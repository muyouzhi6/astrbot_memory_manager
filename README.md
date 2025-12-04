# 🧠 AstrBot Memory Manager (智能记忆管家)

> 为您的 AstrBot 赋予持久化、连贯的长期记忆能力。

[![AstrBot Plugin](https://img.shields.io/badge/AstrBot-Plugin-purple?style=flat-square)](https://github.com/AstrBotDevs/AstrBot)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

**AstrBot Memory Manager** 是一个为 AstrBot 设计的高级记忆管理插件。它通过引入双层缓冲机制（Buffer/Pending）和基于 LLM 的自动总结算法，解决了传统 Bot "聊完就忘" 的痛点，实现了真正的长期记忆。

## ✨ 核心特性

*   **🧠 双层缓冲机制**: 采用 `Buffer` (即时缓存) 和 `Pending` (待处理区) 双层架构，确保在高并发群聊场景下消息不丢失，总结过程原子化。
*   **🔄 自动总结与归档**:
    *   **短期记忆**: 当消息达到阈值（默认50条），自动触发 LLM 总结，提炼关键信息。
    *   **长期记忆**: 每日凌晨自动将前一天的短期摘要合并压缩，形成永久的长期记忆。
*   **💉 智能上下文注入**: 在 AstrBot 调用 LLM 回复用户之前，自动检索并注入相关的最近摘要和重要约定，让 Bot 说话更"懂"你。
*   **📝 重要事项管理**: 支持通过指令 `/memory important` 手动添加、删除、修改重要信息（如群规、用户偏好），最高优先级注入。
*   **🛡️ 鲁棒性设计**: 
    *   LLM 调用失败自动回滚，防止数据丢失。
    *   自动清理过期摘要，防止上下文无限膨胀。
    *   支持自定义 LLM (OpenAI Compatible) 或复用 AstrBot 内置 Provider。

## 📦 安装

1.  将本插件仓库克隆到 AstrBot 的 `data/plugins` 目录：
    ```bash
    cd AstrBot/data/plugins
    git clone https://github.com/muyouzhi6/astrbot_memory_manager.git
    ```
2.  安装依赖：
    ```bash
    pip install -r astrbot_memory_manager/requirements.txt
    ```
3.  重启 AstrBot。

## ⚙️ 配置说明

您可以在 AstrBot 的 WebUI 面板中配置此插件，或直接修改配置项。

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `trigger_threshold` | int | `50` | **触发阈值**：当缓冲区消息数达到此值时，触发自动总结。 |
| `max_buffer_size` | int | `1000` | **最大缓冲**：超过此限制将触发紧急截断（保留最近100条），防止内存溢出。 |
| `reserve_count` | int | `10` | **保留条数**：总结时，保留最近 N 条消息在缓冲区，仅总结旧消息，保持对话连贯性。 |
| `archive_time` | string | `"03:00"` | **归档时间**：每日执行长期记忆合并和清理的时间 (HH:MM)。 |
| `max_retention_days` | int | `30` | **保留天数**：每日摘要保留的最大天数，过期后将被清理或压缩。 |
| `injection_count` | int | `3` | **注入数量**：每次回复时，向 LLM 注入最近 N 条摘要。 |
| `enable_auto_summary` | bool | `true` | **自动总结**：是否启用自动总结功能。 |
| `llm_api_base` | string | `""` | (可选) 自定义 LLM API 地址。留空则使用 AstrBot 默认 Provider。 |
| `llm_api_key` | password | `""` | (可选) 自定义 LLM API Key。 |
| `llm_model_name` | string | `""` | (可选) 自定义 LLM 模型名称 (如 gpt-4o)。 |
| `custom_short_term_prompt` | string | `""` | (可选) 自定义短期总结提示词。使用 `{text}` 代表对话内容。 |
| `custom_long_term_prompt` | string | `""` | (可选) 自定义长期归档提示词。使用 `{text}` (摘要集合) 和 `{date_range}`。 |

## 💻 指令列表

所有指令均以 `/memory` 开头。

| 指令 | 参数 | 说明 |
| :--- | :--- | :--- |
| `/memory view` | 无 | 查看当前会话的最近 5 条记忆摘要。 |
| `/memory view_long` | 无 | 查看当前会话的长期记忆归档。 |
| `/memory important add` | `<内容>` | 添加一条重要事项（如：用户别名、群规）。 |
| `/memory important view` | 无 | 查看所有重要事项。 |
| `/memory important del` | `<序号>` | 删除指定序号的重要事项。 |
| `/memory important edit` | `<序号> <新内容>` | 修改指定序号的重要事项。 |
| `/memory stats` | 无 | 查看当前缓冲、摘要和长期记忆的统计信息。 |
| `/memory clear_buffer` | 无 | [慎用] 清空当前消息缓冲区。 |
| `/memory force_summary` | 无 | [调试] 强制立即执行一次总结。 |
| `/memory export` | 无 | 导出当前会话的所有记忆数据为 JSON 文件。 |
| `/mem_show_t2i` | 无 | 专门展示与文生图（绘图）相关的记忆。 |

## 🔧 运行原理

### 1. 消息缓冲 (Message Buffering)
插件监听所有聊天消息，将其暂存入 `Buffer`。
- **保护机制**: 当 `Buffer` 大小超过 `max_buffer_size` 时，触发紧急截断，仅保留最近 100 条，防止内存泄露。

### 2. 自动总结 (Auto Summary)
当 `Buffer` 大小达到 `trigger_threshold` (默认50) 时：
1.  **锁定**: 锁定处理线程。
2.  **移动 (Move)**: 将前 `Total - reserve_count` 条消息从 `Buffer` 移动到 `Pending Buffer`。
3.  **生成 (Generate)**: 调用 LLM 对 `Pending Buffer` 中的消息生成摘要。
4.  **提交 (Commit)**: 
    - 若成功：将摘要存入 `Daily Summaries`，清空 `Pending Buffer`。
    - 若失败：将 `Pending Buffer` 的消息退回 `Buffer` 头部 (Rollback)，等待下次重试。

### 3. 上下文注入 (Context Injection)
在 AstrBot 处理 LLM 请求前 (`on_llm_request`)：
1.  检索最近的 `injection_count` 条短期摘要。
2.  检索所有 `Important Info`。
3.  将这些信息拼接并追加到 System Prompt 中，使 LLM 能够感知历史上下文。

### 4. 每日归档 (Daily Archiver)
每日指定时间（默认 03:00）：
1.  强制总结当前 `Buffer` 中的剩余消息。
2.  检查超过 `max_retention_days` 的短期摘要。
3.  调用 LLM 将这些过期摘要合并为一个"长期记忆" (Long Term Memory)。
4.  清理旧的短期摘要，释放空间。

## ❓ 常见问题 (Q&A)

**Q: 为什么我的 Bot 没有记忆？**
A: 
1. 检查 `enable_auto_summary` 是否开启。
2. 检查消息量是否达到 `trigger_threshold`。
3. 使用 `/memory stats` 查看缓冲区状态。
4. 检查后台日志是否有 LLM 调用失败的错误。

**Q: 支持哪些 LLM？**
A: 支持所有 OpenAI 兼容接口的 LLM。如果不配置 `llm_api_base`，默认复用 AstrBot 当前配置的 Provider（如 GPT, Claude, DeepSeek 等）。

**Q: 数据存储在哪里？**
A: 数据以 JSON 格式存储在 `data/plugins/astrbot_memory_manager/memories/` 目录下，每个会话一个文件。

**Q: 如何备份记忆？**
A: 您可以直接备份 `memories` 目录，或使用 `/memory export` 指令导出 JSON。

---
**Author**: [木有知](https://github.com/muyouzhi6)
**Repo**: [https://github.com/muyouzhi6/astrbot_memory_manager](https://github.com/muyouzhi6/astrbot_memory_manager)