import asyncio
import os
import time
import json
from typing import List

from astrbot.api.star import Star, Context
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api import logger, AstrBotConfig, llm_tool
from astrbot.core.provider.entities import ProviderRequest
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
except ImportError:
    AsyncIOScheduler = None

from .memory_manager import MemoryManager

class Main(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # Use context.get_data_dir() for data persistence
        self.data_dir = context.get_data_dir()
        self.memory_manager = MemoryManager(context, self.data_dir)
        
        # 启动后台总结任务
        self.summary_task = asyncio.create_task(self._summary_worker())
        
        # self.processing_lock removed in favor of MemoryManager's session_locks

        # 启动定时整理任务 (Archiver)
        self.setup_scheduler()

        # Initialize LLM Client
        self.llm_client = None
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize custom LLM client if configured"""
        api_base = self.config.get("llm_api_base")
        api_key = self.config.get("llm_api_key")
        
        if api_base and api_key:
            if AsyncOpenAI:
                try:
                    self.llm_client = AsyncOpenAI(api_key=api_key, base_url=api_base)
                    logger.info(f"[MemoryManager] Custom LLM client initialized with base: {api_base}")
                except Exception as e:
                    logger.error(f"[MemoryManager] Failed to initialize Custom LLM client: {e}")
            else:
                logger.error("[MemoryManager] openai package not found. Please install it to use custom LLM.")
        else:
            logger.info("[MemoryManager] No custom LLM config found, will use AstrBot default provider.")

    async def _call_llm(self, prompt: str) -> str:
        """Unified LLM call interface with Retry"""
        retries = 3
        base_delay = 2
        
        for attempt in range(retries):
            try:
                # 1. Try Custom LLM
                if self.llm_client:
                    model_name = self.config.get("llm_model_name", "gpt-3.5-turbo")
                    response = await self.llm_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    content = response.choices[0].message.content
                    return content if content is not None else ""
                
                # 2. Use AstrBot Default Provider
                providers = self.context.get_all_providers()
                if not providers:
                    logger.warning("[MemoryManager] No AstrBot LLM provider available.")
                    return ""
                
                provider = providers[0]
                response = await provider.text_chat(prompt=prompt, session=None) # type: ignore
                if response and response.completion_text:
                    return response.completion_text
                else:
                    raise ValueError("Empty response from provider")

            except Exception as e:
                logger.error(f"[MemoryManager] LLM call failed (Attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"[MemoryManager] Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("[MemoryManager] Max retries reached. Giving up.")
                    return ""
        return ""

    def setup_scheduler(self):
        archive_time = self.config.get("archive_time", "03:00")
        try:
            hour, minute = map(int, archive_time.split(":"))
        except ValueError:
            logger.error(f"[MemoryManager] Invalid archive_time format: {archive_time}, using default 03:00")
            hour, minute = 3, 0

        if AsyncIOScheduler:
            self.scheduler = AsyncIOScheduler()
            self.scheduler.add_job(self._daily_archive, 'cron', hour=hour, minute=minute)
            self.scheduler.start()
            logger.info(f"[MemoryManager] Daily archive scheduled at {hour:02d}:{minute:02d}")
        else:
            logger.warning("[MemoryManager] apscheduler not found, daily archive disabled.")
            self.scheduler = None

    async def _summary_worker(self):
        """后台任务：监控所有会话的 buffer 并触发总结"""
        while True:
            try:
                # Iterate over all active sessions
                session_ids = self.memory_manager.get_all_session_ids()
                threshold = self.config.get("trigger_threshold", 50)
                
                for session_id in session_ids:
                    buffer_size = await self.memory_manager.get_buffer_size(session_id)
                    if buffer_size >= threshold:
                        logger.info(f"[MemoryManager] Session {session_id}: Buffer size {buffer_size} >= {threshold}, triggering summary.")
                        # Create a task for each session summary to run concurrently
                        asyncio.create_task(self.perform_summary(session_id))
                
                await asyncio.sleep(60) # 每分钟检查一次
            except Exception as e:
                logger.error(f"[MemoryManager] Summary worker error: {e}")
                await asyncio.sleep(60)

    async def perform_summary(self, session_id: str, force: bool = False):
        """执行总结逻辑 (Per Session)"""
        
        if not self.config.get("enable_auto_summary", True) and not force:
            return

        reserve_count = 0 if force else self.config.get("reserve_count", 10)
        custom_prompt = self.config.get("custom_short_term_prompt", "")
        
        # 1. Get buffer content (snapshot)
        current_buffer = await self.memory_manager.get_buffer_content(session_id)
        total_count = len(current_buffer)
        
        if total_count <= reserve_count and not force:
            return

        summarize_count = total_count - reserve_count
        if summarize_count <= 0:
            return

        # 2. Move to pending (Atomic Step 1)
        messages_to_summarize = await self.memory_manager.move_to_pending(session_id, summarize_count)
        if not messages_to_summarize:
            return

        # 3. Build Prompt
        messages_text = "\n".join([f"{msg.get('sender')}: {msg.get('content')}" for msg in messages_to_summarize])
        
        if custom_prompt and "{text}" in custom_prompt:
            prompt = custom_prompt.replace("{text}", messages_text)
        else:
            prompt = f"""请总结以下群聊消息。
要求：
1. 总结需包含：关键事件、重要决策、用户提到的待办事项或意图。
2. 忽略无意义的闲聊、表情包和重复信息。
3. 保持客观、准确，不要臆造内容。
4. 字数控制在 500 字以内。
5. 格式清晰，分点列出。

消息记录：
{messages_text}
"""

        # 4. Call LLM for Summary
        summary_task = asyncio.create_task(self._call_llm(prompt))
        
        # 4.1 Call LLM for Structured Extraction (Parallel)
        extract_task = None
        if self.config.get("enable_structured_extraction", True):
            schema = self.config.get("structured_extraction_schema", "")
            if not schema:
                schema = """{
    "user_profiles": {
        "type": "list",
        "items": {
            "nickname": "string",
            "location": "string",
            "preferences": ["string"],
            "relationship_status": "string"
        }
    },
    "tasks": {
        "type": "list",
        "items": {
            "content": "string",
            "assignee": "string",
            "deadline": "string"
        }
    }
}"""
            extract_prompt = f"""请从以下对话中提取结构化信息。
目标 Schema:
{schema}

对话内容:
{messages_text}

要求：
1. 仅输出符合 Schema 的 JSON 字符串，不要包含其他说明文字。
2. 如果没有相关信息，返回空 JSON 对象 {{}}。
"""
            extract_task = asyncio.create_task(self._call_llm(extract_prompt))

        # Wait for results
        summary = await summary_task
        structured_info_str = await extract_task if extract_task else None

        # 5. Commit or Rollback (Atomic Step 2)
        if summary:
            await self.memory_manager.commit_pending(session_id, summary)
            logger.info(f"[MemoryManager] Session {session_id}: Summary committed. Processed {len(messages_to_summarize)} messages.")
            
            # Process Structured Info
            if structured_info_str:
                try:
                    structured_info = json.loads(structured_info_str)
                    if structured_info:
                        await self.memory_manager.update_structured_data(session_id, structured_info)
                        logger.info(f"[MemoryManager] Session {session_id}: Structured info updated.")
                except json.JSONDecodeError:
                    logger.warning(f"[MemoryManager] Failed to parse structured info JSON.")
            
            # 6. Check Threshold and Archive
            threshold = self.config.get("daily_summary_threshold", 20)
            if threshold > 0:
                current_summaries = await self.memory_manager.get_all_daily_summaries(session_id)
                if len(current_summaries) >= threshold:
                    logger.info(f"[MemoryManager] Session {session_id}: Summary count {len(current_summaries)} >= {threshold}. Triggering archive.")
                    asyncio.create_task(self.archive_summaries(session_id))
        else:
            await self.memory_manager.rollback_pending(session_id)
            logger.warning(f"[MemoryManager] Session {session_id}: Summary failed (empty response), rolled back.")

    async def archive_summaries(self, session_id: str):
        """将短期摘要归档为长期记忆"""
        session = await self.memory_manager.get_session(session_id)
        summaries = session.daily_summaries
        if not summaries:
            return

        summaries_text = "\n".join([f"{s['date']}: {s['content']}" for s in summaries])
        date_range = f"{summaries[0]['date']} - {summaries[-1]['date']}"
        # Get timestamps to remove specifically these summaries later
        # Ensure timestamps are floats
        timestamps_to_remove: List[float] = [
            float(s['timestamp']) for s in summaries
            if s.get('timestamp') is not None
        ]
        
        custom_prompt = self.config.get("custom_long_term_prompt", "")
        if custom_prompt and "{text}" in custom_prompt:
            prompt = custom_prompt.replace("{text}", summaries_text).replace("{date_range}", date_range)
        else:
            prompt = f"以下是 {date_range} 期间的对话摘要：\n{summaries_text}\n\n请将这些摘要合并为一个连贯的、高度概括的长期记忆。重点保留关键事件、重要决策和用户偏好，忽略琐碎细节。"
        
        long_term_summary = await self._call_llm(prompt)
        
        if long_term_summary:
            await self.memory_manager.add_long_term_memory(session_id, long_term_summary, date_range)
            # Remove only the summaries that were included in this archive process
            # This prevents deleting new summaries that arrived during LLM processing
            if timestamps_to_remove:
                await self.memory_manager.remove_summaries_by_timestamp(session_id, timestamps_to_remove)
            else:
                # Fallback if timestamps are missing (shouldn't happen with new data)
                await self.memory_manager.clear_all_daily_summaries(session_id)
            logger.info(f"[MemoryManager] Session {session_id}: Archived {len(summaries)} summaries into long-term memory.")

    async def _daily_archive(self):
        """每日整理任务 (All Sessions)"""
        logger.info("[MemoryManager] Starting daily archive task...")
        
        session_ids = self.memory_manager.get_all_session_ids()
        for session_id in session_ids:
            # 1. Force summary remaining buffer
            await self.perform_summary(session_id, force=True)
            
            # 2. Generate Daily Recap (New V2 Feature)
            # Get summaries from the last 24 hours
            yesterday_start = time.time() - 86400
            session = await self.memory_manager.get_session(session_id)
            daily_summaries_to_recap = [
                s for s in session.daily_summaries
                if s.get('timestamp', 0) >= yesterday_start
            ]
            
            if daily_summaries_to_recap:
                logger.info(f"[MemoryManager] Session {session_id}: Generating Daily Recap for {len(daily_summaries_to_recap)} summaries.")
                summaries_text = "\n".join([f"{s.get('date')}: {s.get('content')}" for s in daily_summaries_to_recap])
                date_str = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400)) # Yesterday's date
                
                prompt = f"""请为以下昨日的对话片段生成一份结构化的日报。
日期：{date_str}

内容片段：
{summaries_text}

要求：
1. 提炼关键信息，包括核心话题、达成的一致、未解决的问题。
2. 忽略琐碎的闲聊。
3. 如果有具体的任务或待办，请单独列出。
4. 输出格式：
   - 【核心话题】...
   - 【关键结论】...
   - 【待办/遗留】...
"""
                recap = await self._call_llm(prompt)
                if recap:
                    await self.memory_manager.add_long_term_memory(session_id, recap, f"日报 {date_str}")
                    logger.info(f"[MemoryManager] Session {session_id}: Daily Recap generated.")

            # 3. Clean up/Compress old summaries
            max_days = self.config.get("max_retention_days", 30)
            expired_summaries = await self.memory_manager.get_expired_summaries(session_id, max_days)
            
            if expired_summaries:
                logger.info(f"[MemoryManager] Session {session_id}: Found {len(expired_summaries)} expired summaries. Compressing...")
                await self.compress_memories(session_id, expired_summaries)
                await self.memory_manager.cleanup_summaries(session_id, max_days)
        
        logger.info("[MemoryManager] Daily archive task completed.")

    async def compress_memories(self, session_id: str, summaries: List[dict]):
        """递归压缩长期记忆"""
        if not summaries:
            return

        BATCH_SIZE = 10
        long_term_summary = ""
        date_range = f"{summaries[0].get('date', '?')} - {summaries[-1].get('date', '?')}"

        if len(summaries) > BATCH_SIZE:
            # Recursive Batching
            chunks = [summaries[i:i + BATCH_SIZE] for i in range(0, len(summaries), BATCH_SIZE)]
            logger.info(f"[MemoryManager] Session {session_id}: Compressing {len(summaries)} summaries in {len(chunks)} chunks.")
            
            intermediate_results = []
            for chunk in chunks:
                chunk_text = "\n".join([f"{s.get('date', '?')}: {s.get('content', '')}" for s in chunk])
                prompt = f"请将以下对话摘要合并为一个简练的概括：\n{chunk_text}"
                res = await self._call_llm(prompt)
                if res:
                    intermediate_results.append(res)
            
            final_text = "\n".join([f"片段{i+1}: {text}" for i, text in enumerate(intermediate_results)])
            final_prompt = f"以下是 {date_range} 期间的多个记忆片段，请将它们整合成一个完整的长期记忆叙述：\n{final_text}"
            long_term_summary = await self._call_llm(final_prompt)
            
        else:
            # Direct compression
            summaries_text = "\n".join([f"{s.get('date', '?')}: {s.get('content', '')}" for s in summaries])
            
            custom_prompt = self.config.get("custom_long_term_prompt", "")
            if custom_prompt and "{text}" in custom_prompt:
                prompt = custom_prompt.replace("{text}", summaries_text).replace("{date_range}", date_range)
            else:
                prompt = f"以下是 {date_range} 期间的每日对话摘要：\n{summaries_text}\n\n请将这些零散的摘要合并为一个连贯的、高度概括的长期记忆。重点保留关键事件、重要决策和用户偏好，忽略琐碎细节。"
            
            long_term_summary = await self._call_llm(prompt)
        
        if long_term_summary:
            await self.memory_manager.add_long_term_memory(session_id, long_term_summary, date_range)
            logger.info(f"[MemoryManager] Session {session_id}: Compressed memories for {date_range}.")

    @filter.on_astrbot_loaded()
    async def on_start(self, event: AstrMessageEvent):
        logger.info("[MemoryManager] Plugin started.")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听所有消息并存入 buffer"""
        if event.message_str: # 忽略空消息
            session_id = event.unified_msg_origin
            msg_data = {
                "sender": event.get_sender_name(),
                "content": event.message_str,
                "time": time.time()
            }
            max_buffer = self.config.get("max_buffer_size", 1000)
            await self.memory_manager.add_message(session_id, msg_data, max_buffer)

    @filter.on_llm_request()
    async def inject_memory(self, event: AstrMessageEvent, req: ProviderRequest):
        """在 LLM 请求前注入记忆"""
        if not self.config.get("enable_auto_summary", True):
            return

        session_id = event.unified_msg_origin
        
        # 1. Get Structured Info (New)
        structured_data = await self.memory_manager.get_structured_data(session_id)
        
        # 2. Get Important Info
        important_infos = await self.memory_manager.get_important_info(session_id)
        
        # 3. Get Summaries with Token Control
        inject_all = self.config.get("inject_all_summaries", False)
        injection_count = self.config.get("injection_count", 3)
        max_tokens = self.config.get("max_injection_tokens", 2000)
        
        summaries = []
        if inject_all:
            all_summaries = await self.memory_manager.get_all_daily_summaries(session_id)
            # Reverse to keep recent first, then slice by tokens
            current_tokens = 0
            for s in reversed(all_summaries):
                content = s.get("content", "") if isinstance(s, dict) else str(s)
                tokens = self.memory_manager.count_tokens(content)
                if current_tokens + tokens > max_tokens:
                    break
                summaries.insert(0, s)
                current_tokens += tokens
        else:
            summaries = await self.memory_manager.get_recent_summaries(session_id, injection_count)
        
        # Build Memory Text
        memory_parts = []
        
        if structured_data:
            info_str = json.dumps(structured_data, ensure_ascii=False)
            memory_parts.append(f"【用户画像/结构化信息】\n{info_str}")

        if important_infos:
            memory_parts.append("【重要约定/信息】\n" + "\n".join([f"- {info}" for info in important_infos]))
            
        if summaries:
            title = "【对话摘要历史】" if inject_all else "【最近对话摘要】"
            memory_parts.append(f"{title}\n" + "\n".join([f"- {s}" for s in summaries]))
            
        if memory_parts:
            memory_text = "\n\n".join(memory_parts)
            system_prompt_appendix = f"\n\n=== 记忆注入 ===\n{memory_text}\n================\n请在回答时参考以上记忆信息。"
            current_sys_prompt = req.system_prompt or ""
            req.system_prompt = current_sys_prompt + system_prompt_appendix

    @llm_tool(name="search_chat_history")
    async def search_chat_history(self, event: AstrMessageEvent, keywords: str, days: int = 30):
        """
        当默认提供的对话摘要信息不足以回答用户问题，或者用户明确询问过去的某个具体事件时，使用此工具搜索历史记忆。
        此工具会搜索所有的每日总结、长期记忆和重要事项。

        Args:
            keywords (string): 搜索关键词，多个关键词用空格分隔。
            days (number): 搜索最近多少天的记录，默认为 30 天。如果需要搜索更早的记录，请填入更大的数字。
        """
        session_id = event.unified_msg_origin
        
        results = await self.memory_manager.search_memories(session_id, keywords, days)
        
        if not results:
            return "未找到相关历史记忆。"
        
        # Limit results to avoid overflowing context
        return "找到以下相关历史记忆：\n" + "\n".join(results[-10:])

    @filter.command("memory")
    async def memory_cmd(self, event: AstrMessageEvent, action: str = "view", *args):
        """
        记忆管理指令
        /memory view - 查看最近记忆
        /memory view_long - 查看长期记忆
        /memory important [add/del/view] - 管理重要事项
        /memory fill_form <key> <value> - 手动填表（更新结构化信息）
        /memory view_form - 查看结构化信息
        /memory clear_buffer - 清空当前缓冲
        /memory force_summary - 强制执行总结
        """
        session_id = event.unified_msg_origin
        
        if action == "view":
            summaries = await self.memory_manager.get_recent_summaries(session_id, 5)
            if not summaries:
                yield event.plain_result("暂无记忆摘要。")
            else:
                text = "【最近记忆摘要】\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(summaries)])
                yield event.plain_result(text)

        elif action == "view_long":
            long_term = await self.memory_manager.get_long_term_memories(session_id)
            if not long_term:
                yield event.plain_result("暂无长期记忆。")
            else:
                text_list = []
                for item in long_term[-5:]:
                    text_list.append(f"[{item['date_range']}]\n{item['content']}")
                text = "【长期记忆归档】\n\n" + "\n\n".join(text_list)
                yield event.plain_result(text)
        
        elif action == "important":
            sub_action = args[0] if args else "view"
            if sub_action == "add":
                if len(args) < 2:
                    yield event.plain_result("请提供重要事项内容。")
                    return
                content = " ".join(args[1:])
                await self.memory_manager.add_important_info(session_id, content)
                yield event.plain_result("已添加重要事项。")
            elif sub_action == "del":
                if len(args) < 2:
                    yield event.plain_result("请提供要删除的事项序号（从1开始）。")
                    return
                try:
                    index = int(args[1]) - 1
                    if await self.memory_manager.remove_important_info(session_id, index):
                        yield event.plain_result("删除成功。")
                    else:
                        yield event.plain_result("删除失败，序号无效。")
                except ValueError:
                    yield event.plain_result("序号必须是数字。")
            elif sub_action == "edit":
                if len(args) < 3:
                    yield event.plain_result("请提供要编辑的事项序号和新内容。例如：/memory important edit 1 新内容")
                    return
                try:
                    index = int(args[1]) - 1
                    new_content = " ".join(args[2:])
                    if await self.memory_manager.edit_important_info(session_id, index, new_content):
                        yield event.plain_result("修改成功。")
                    else:
                        yield event.plain_result("修改失败，序号无效。")
                except ValueError:
                    yield event.plain_result("序号必须是数字。")
            elif sub_action == "view":
                infos = await self.memory_manager.get_important_info(session_id)
                if not infos:
                    yield event.plain_result("暂无重要事项。")
                else:
                    text = "【重要事项】\n" + "\n".join([f"{i+1}. {info}" for i, info in enumerate(infos)])
                    yield event.plain_result(text)
            else:
                yield event.plain_result("未知子指令。可用: add, del, edit, view")

        elif action == "fill_form":
            if len(args) < 2:
                yield event.plain_result("请提供键和值。例如：/memory fill_form location 北京")
                return
            key = args[0]
            value = " ".join(args[1:])
            try:
                await self.memory_manager.update_structured_data(session_id, {key: value})
                yield event.plain_result(f"已更新结构化信息: {key} = {value}")
            except Exception as e:
                logger.error(f"[MemoryManager] Failed to update structured data: {e}")
                yield event.plain_result("更新失败，请稍后重试。")

        elif action == "view_form":
            data = await self.memory_manager.get_structured_data(session_id)
            if not data:
                yield event.plain_result("暂无结构化信息。")
            else:
                text = "【结构化信息】\n" + json.dumps(data, ensure_ascii=False, indent=2)
                yield event.plain_result(text)

        elif action == "clear_buffer":
            await self.memory_manager.clear_buffer(session_id)
            yield event.plain_result("消息缓冲区已清空。")
            
        elif action == "force_summary":
            yield event.plain_result("正在执行强制总结（包含保留区消息）...")
            try:
                await self.perform_summary(session_id, force=True)
                yield event.plain_result("总结执行完毕。")
            except Exception as e:
                logger.error(f"[MemoryManager] Force summary failed: {e}")
                yield event.plain_result(f"总结执行失败: {e}")

        elif action == "stats":
            try:
                buffer_size = await self.memory_manager.get_buffer_size(session_id)
                session = await self.memory_manager.get_session(session_id)
                pending_size = len(session.pending_buffer)
                summary_count = len(session.daily_summaries)
                long_term_count = len(session.long_term_memory)
                max_buffer = self.config.get("max_buffer_size", 1000)
                
                stats_text = (
                    "【记忆统计】\n"
                    f"Session ID: {session_id}\n"
                    f"Buffer Size: {buffer_size} / {max_buffer}\n"
                    f"Pending Buffer: {pending_size}\n"
                    f"Daily Summaries: {summary_count}\n"
                    f"Long Term Memories: {long_term_count}"
                )
                yield event.plain_result(stats_text)
            except Exception as e:
                logger.error(f"[MemoryManager] Stats error: {e}")
                yield event.plain_result(f"获取统计信息失败: {e}")

        elif action == "export":
            session = await self.memory_manager.get_session(session_id)
            data = session.to_dict()
            try:
                # Create export directory if not exists
                export_dir = os.path.join(self.data_dir, "exports")
                if not os.path.exists(export_dir):
                    os.makedirs(export_dir, exist_ok=True)
                
                # Sanitize filename
                safe_sid = session_id.replace(":", "_")
                filename = f"memory_export_{safe_sid}_{int(time.time())}.json"
                filepath = os.path.join(export_dir, filename)
                
                # Security check: ensure filepath is within export_dir
                if not os.path.abspath(filepath).startswith(os.path.abspath(export_dir)):
                     yield event.plain_result("导出失败: 非法的文件路径。")
                     return

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    
                yield event.plain_result(f"记忆已导出到数据目录: exports/{filename}\n(由于安全限制，请联系管理员在服务器获取文件)")
            except Exception as e:
                yield event.plain_result(f"导出失败: {e}")
            
        else:
            yield event.plain_result("未知指令。可用: view, view_long, important, fill_form, view_form, clear_buffer, force_summary, stats, export")

    @filter.command("mem_show_t2i")
    async def show_t2i_memory(self, event: AstrMessageEvent):
        """展示文生图相关记忆"""
        session_id = event.unified_msg_origin
        # 简单实现：过滤包含 "绘图", "画图", "生成图片" 等关键词的摘要
        keywords = ["绘图", "画图", "生成图片", "文生图", "drawing", "image generation"]
        
        # Accessing internal list directly for filtering
        session = await self.memory_manager.get_session(session_id)
        relevant_summaries = []
        
        # Filter daily summaries
        for s in session.daily_summaries:
            if any(k in s.get('content', '') for k in keywords):
                relevant_summaries.append(f"[{s.get('date', 'Unknown Date')}] {s.get('content', '')}")
                
        # Filter long term memories
        for s in session.long_term_memory:
            if any(k in s.get('content', '') for k in keywords):
                relevant_summaries.append(f"[{s.get('date_range', 'Unknown Range')}] {s.get('content', '')}")
                
        if not relevant_summaries:
            yield event.plain_result("未找到与文生图相关的记忆。")
        else:
            text = "【文生图相关记忆】\n\n" + "\n\n".join(relevant_summaries[-10:]) # Show last 10
            yield event.plain_result(text)

    async def terminate(self):
        self.summary_task.cancel()
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
        
        # Ensure all data is flushed
        if hasattr(self.memory_manager, 'flush_task'):
            self.memory_manager.flush_task.cancel()
        await self.memory_manager.flush_all_dirty_sessions()
        
        logger.info("[MemoryManager] Plugin terminated.")