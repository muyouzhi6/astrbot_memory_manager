import asyncio
import json
import os
import time
import shutil
import aiofiles
import tiktoken
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
from astrbot.api import logger
from astrbot.api.star import Context

@dataclass
class SessionMemory:
    session_id: str
    buffer: List[Dict[str, Any]] = field(default_factory=list)
    pending_buffer: List[Dict[str, Any]] = field(default_factory=list)
    daily_summaries: List[Dict[str, Any]] = field(default_factory=list)
    long_term_memory: List[Dict[str, Any]] = field(default_factory=list)
    important_info: List[str] = field(default_factory=list)
    structured_data: Dict[str, Any] = field(default_factory=dict) # New: For "填表" feature
    
    # Dirty flag for delayed write
    _dirty: bool = field(default=False, init=False, repr=False)
    _last_save_time: float = field(default=0.0, init=False, repr=False)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "buffer": self.buffer,
            "pending_buffer": self.pending_buffer,
            "daily_summaries": self.daily_summaries,
            "long_term_memory": self.long_term_memory,
            "important_info": self.important_info,
            "structured_data": self.structured_data
        }
    
    def mark_dirty(self):
        self._dirty = True

    def is_dirty(self):
        return self._dirty
        
    def clean(self):
        self._dirty = False
        self._last_save_time = time.time()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            session_id=data.get("session_id", "unknown"),
            buffer=data.get("buffer", []),
            pending_buffer=data.get("pending_buffer", []),
            daily_summaries=data.get("daily_summaries", []),
            long_term_memory=data.get("long_term_memory", []),
            important_info=data.get("important_info", []),
            structured_data=data.get("structured_data", {})
        )

class MemoryManager:
    def __init__(self, context: Context, data_path: str):
        self.context = context
        self.data_path = data_path
        self.base_data_path = os.path.join(data_path, "memories")
        if not os.path.exists(self.base_data_path):
            os.makedirs(self.base_data_path, exist_ok=True)
            
        # Session-level locks for fine-grained concurrency control
        self.session_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.global_lock = asyncio.Lock()
        
        self.active_sessions: Dict[str, SessionMemory] = {}
        self.known_session_ids: Set[str] = self._scan_session_files()
        
        # Start background flush task
        self.flush_task = asyncio.create_task(self._background_flush_worker())
        
        self.encoder = None

    async def initialize(self):
        """Async initialization"""
        await self._migrate_legacy_data(self.data_path)
        
        # Init tiktoken encoder in executor to avoid blocking
        try:
            loop = asyncio.get_running_loop()
            self.encoder = await loop.run_in_executor(None, tiktoken.get_encoding, "cl100k_base")
        except Exception as e:
            logger.warning(f"[MemoryManager] Failed to load tiktoken encoding: {e}. Fallback to char estimation.")
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception:
                return len(text)
        return len(text)

    async def _background_flush_worker(self):
        """Background task to flush dirty sessions periodically"""
        FLUSH_INTERVAL = 10 # seconds
        while True:
            try:
                await asyncio.sleep(FLUSH_INTERVAL)
                await self.flush_all_dirty_sessions()
            except asyncio.CancelledError:
                # Ensure we flush on cancel
                await self.flush_all_dirty_sessions()
                break
            except Exception as e:
                logger.error(f"[MemoryManager] Flush worker error: {e}")

    async def flush_all_dirty_sessions(self):
        """Save all dirty sessions to disk"""
        # Copy keys to avoid runtime error during iteration if dict changes
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            session = self.active_sessions.get(session_id)
            if session and session.is_dirty():
                async with self._get_lock(session_id):
                    if session.is_dirty(): # Check again inside lock
                        await self.save_session(session)

    def _scan_session_files(self) -> Set[str]:
        ids = set()
        if not os.path.exists(self.base_data_path):
            return ids
            
        for f in os.listdir(self.base_data_path):
            if f.endswith(".json"):
                sid = f[:-5].replace("_COLON_", ":")
                ids.add(sid)
        return ids

    def _get_file_path(self, session_id: str) -> str:
        safe_sid = session_id.replace(":", "_COLON_")
        return os.path.join(self.base_data_path, f"{safe_sid}.json")

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        return self.session_locks[session_id]

    async def _migrate_legacy_data(self, old_data_path: str):
        v2_file = os.path.join(old_data_path, "memory_data_v2.json")
        v1_file = os.path.join(old_data_path, "memory_data.json")
        
        source_file = None
        if os.path.exists(v2_file):
            source_file = v2_file
        elif os.path.exists(v1_file):
            source_file = v1_file
            
        if source_file:
            logger.info(f"[MemoryManager] Migrating data from {source_file} to new structure...")
            try:
                # Run sync IO in executor
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._do_migration_sync, source_file, v1_file)
                logger.info("[MemoryManager] Migration completed.")
            except Exception as e:
                logger.error(f"[MemoryManager] Migration failed: {e}")

    def _do_migration_sync(self, source_file, v1_file):
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if source_file == v1_file:
            sid = "default"
            sm = SessionMemory.from_dict(data)
            sm.session_id = sid
            self._save_session_sync(sm)
            self.known_session_ids.add(sid)
        else:
            for sid, sdata in data.items():
                sm = SessionMemory.from_dict(sdata)
                self._save_session_sync(sm)
                self.known_session_ids.add(sid)
        
        shutil.move(source_file, source_file + ".bak")

    def _save_session_sync(self, session: SessionMemory):
        file_path = self._get_file_path(session.session_id)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=4)

    async def _load_session(self, session_id: str) -> SessionMemory:
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
            
        file_path = self._get_file_path(session_id)
        if os.path.exists(file_path):
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    session = SessionMemory.from_dict(data)
                    
                    if session.pending_buffer:
                        logger.warning(f"[MemoryManager] Session {session_id}: Found {len(session.pending_buffer)} pending messages on load. Rolling back.")
                        session.buffer = session.pending_buffer + session.buffer
                        session.pending_buffer = []
                    
                    self.active_sessions[session_id] = session
                    self.known_session_ids.add(session_id)
                    return session
            except Exception as e:
                logger.error(f"[MemoryManager] Failed to load session {session_id}: {e}")
        
        session = SessionMemory(session_id=session_id)
        self.active_sessions[session_id] = session
        self.known_session_ids.add(session_id)
        return session

    async def save_session(self, session: SessionMemory):
        file_path = self._get_file_path(session.session_id)
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                content = json.dumps(session.to_dict(), ensure_ascii=False, indent=4)
                await f.write(content)
            session.clean()
        except Exception as e:
            logger.error(f"[MemoryManager] Failed to save session {session.session_id}: {e}")

    async def get_session(self, session_id: str) -> SessionMemory:
        return await self._load_session(session_id)

    async def add_message(self, session_id: str, message: Dict[str, Any], max_buffer_size: int = 1000):
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            session.buffer.append(message)
            
            # Buffer protection mechanism
            # Simple length check, ideally should be token check
            if len(session.buffer) > max_buffer_size:
                keep_count = 100
                logger.warning(f"[MemoryManager] Session {session_id}: Buffer size {len(session.buffer)} exceeded limit {max_buffer_size}. Truncating to last {keep_count} messages.")
                session.buffer = session.buffer[-keep_count:]
                
            session.mark_dirty() # Mark as dirty instead of saving immediately

    async def move_to_pending(self, session_id: str, count: int) -> List[Dict[str, Any]]:
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            if not session.buffer:
                return []
            
            count = min(count, len(session.buffer))
            messages = session.buffer[:count]
            session.buffer = session.buffer[count:]
            session.pending_buffer.extend(messages)
            session.mark_dirty() # Mark dirty
            await self.save_session(session) # Critical operation, force save
            return messages

    async def commit_pending(self, session_id: str, summary: str, date_str: Optional[str] = None):
        if date_str is None:
            date_str = time.strftime("%Y-%m-%d", time.localtime())
            
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            session.pending_buffer = []
            session.daily_summaries.append({
                "date": date_str,
                "content": summary,
                "timestamp": time.time()
            })
            session.mark_dirty()
            await self.save_session(session) # Critical

    async def rollback_pending(self, session_id: str):
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            if session.pending_buffer:
                session.buffer = session.pending_buffer + session.buffer
                session.pending_buffer = []
                session.mark_dirty()
                await self.save_session(session) # Critical

    async def clear_buffer(self, session_id: str):
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            session.buffer = []
            session.pending_buffer = []
            session.mark_dirty()
            await self.save_session(session) # Immediate effect

    async def add_long_term_memory(self, session_id: str, summary: str, date_range: str):
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            session.long_term_memory.append({
                "date_range": date_range,
                "content": summary,
                "timestamp": time.time()
            })
            session.mark_dirty()
            await self.save_session(session) # Critical

    async def add_important_info(self, session_id: str, info: str):
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            session.important_info.append(info)
            session.mark_dirty()
            await self.save_session(session) # Critical

    async def edit_important_info(self, session_id: str, index: int, new_info: str) -> bool:
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            if 0 <= index < len(session.important_info):
                session.important_info[index] = new_info
                session.mark_dirty()
                await self.save_session(session) # Critical
                return True
            return False

    async def remove_important_info(self, session_id: str, index: int) -> bool:
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            if 0 <= index < len(session.important_info):
                session.important_info.pop(index)
                session.mark_dirty()
                await self.save_session(session) # Critical
                return True
            return False

    async def update_structured_data(self, session_id: str, data: Dict[str, Any]):
        """New: Batch update structured data"""
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            session.structured_data.update(data)
            session.mark_dirty()
            # Don't force save for structured data updates, let it flush

    async def get_structured_data(self, session_id: str) -> Dict[str, Any]:
        session = await self.get_session(session_id)
        return session.structured_data

    async def get_buffer_size(self, session_id: str) -> int:
        session = await self.get_session(session_id)
        return len(session.buffer)

    async def get_buffer_content(self, session_id: str) -> List[Dict[str, Any]]:
        session = await self.get_session(session_id)
        return session.buffer

    async def get_recent_summaries(self, session_id: str, count: int = 3) -> List[str]:
        session = await self.get_session(session_id)
        summaries = [item["content"] for item in session.daily_summaries[-count:]]
        if len(summaries) < count:
            remaining = count - len(summaries)
            long_term = [item["content"] for item in session.long_term_memory[-remaining:]]
            summaries = long_term + summaries
        return summaries

    async def get_all_daily_summaries(self, session_id: str) -> List[Dict[str, Any]]:
        """获取所有短期摘要"""
        session = await self.get_session(session_id)
        return session.daily_summaries

    async def clear_all_daily_summaries(self, session_id: str):
        """清空所有短期摘要"""
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            session.daily_summaries = []
            session.mark_dirty()
            await self.save_session(session)

    async def remove_summaries_by_timestamp(self, session_id: str, timestamps: List[float]):
        """根据时间戳删除指定的短期摘要"""
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            ts_set = set(timestamps)
            original_count = len(session.daily_summaries)
            session.daily_summaries = [
                s for s in session.daily_summaries
                if s.get("timestamp") not in ts_set
            ]
            if len(session.daily_summaries) != original_count:
                session.mark_dirty()
                await self.save_session(session)

    async def get_long_term_memories(self, session_id: str) -> List[Dict[str, Any]]:
        session = await self.get_session(session_id)
        return session.long_term_memory

    async def get_important_info(self, session_id: str) -> List[str]:
        session = await self.get_session(session_id)
        return session.important_info

    async def get_expired_summaries(self, session_id: str, max_days: int) -> List[Dict[str, Any]]:
        session = await self.get_session(session_id)
        if max_days <= 0:
            return []
        
        cutoff_time = time.time() - (max_days * 86400)
        return [
            s for s in session.daily_summaries
            if s.get("timestamp", 0) <= cutoff_time
        ]

    async def cleanup_summaries(self, session_id: str, max_days: int):
        if max_days <= 0:
            return
            
        cutoff_time = time.time() - (max_days * 86400)
        async with self._get_lock(session_id):
            session = await self.get_session(session_id)
            original_count = len(session.daily_summaries)
            session.daily_summaries = [
                s for s in session.daily_summaries
                if s.get("timestamp", 0) > cutoff_time
            ]
            new_count = len(session.daily_summaries)
            
            if original_count != new_count:
                logger.info(f"[MemoryManager] Cleaned up {original_count - new_count} old summaries for session {session_id}.")
                session.mark_dirty()
                await self.save_session(session)

    async def search_memories(self, session_id: str, keywords: str, days: int = 30) -> List[str]:
        session = await self.get_session(session_id)
        results = []
        keywords_list = keywords.split()
        
        def match(text):
            if not text: return False
            return any(k in text for k in keywords_list)
            
        cutoff_time = 0
        if days > 0:
            cutoff_time = time.time() - (days * 86400)
        
        # Search Structured Data
        for k, v in session.structured_data.items():
             if match(str(k)) or match(str(v)):
                 results.append(f"[结构化信息] {k}: {v}")

        for s in session.daily_summaries:
            if s.get("timestamp", 0) >= cutoff_time and match(s.get("content", "")):
                results.append(f"[每日总结 {s.get('date')}] {s.get('content')}")
                
        for s in session.long_term_memory:
             if match(s.get("content", "")):
                 results.append(f"[长期记忆 {s.get('date_range')}] {s.get('content')}")

        for info in session.important_info:
            if match(info):
                results.append(f"[重要事项] {info}")
                
        return results
            
    def get_all_session_ids(self) -> List[str]:
        return list(self.known_session_ids)