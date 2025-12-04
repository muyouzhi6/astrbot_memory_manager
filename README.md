# 🧠 AstrBot Memory Manager (智能记忆管家)

> **赋予 AstrBot 真正的长期记忆与自我演化能力**
>
> 基于双缓冲机制与 LLM 自动摘要算法，实现从"碎片化对话"到"结构化记忆"的完整生命周期管理。

[![AstrBot Plugin](https://img.shields.io/badge/AstrBot-Plugin-purple?style=flat-square)](https://github.com/AstrBotDevs/AstrBot)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

**AstrBot Memory Manager** 是一个专家级的记忆管理插件，旨在解决传统 LLM 机器人"聊完就忘"、"上下文有限"的痛点。它像人类的海马体一样，在后台默默地将海量的聊天记录转化为精炼的摘要、结构化的知识和永久的长期记忆。

---

## 🛠️ 核心工作流与原理

本插件的设计灵感来源于人类的记忆形成机制：**短期记忆缓冲** -> **工作记忆加工 (摘要)** -> **长期记忆固化 (归档)**。


### 核心概念解释

1.  **m (trigger_threshold) 与 n (reserve_count)**:
    *   这是为了解决"总结时打断当前对话"的问题。
    *   当积累了 **m** 条消息时，系统会在后台启动总结。
    *   但是，它不会把这 m 条全拿走，而是**保留最近的 n 条**不动（因为这 n 条很可能是当前正在进行的对话，还没结束）。
    *   系统只总结 **m-n** 条旧消息。这样既及时清理了内存，又保证了 Bot 对当前话题的反应不出错。

2.  **填表 (结构化信息提取)**:
    *   在总结对话的同时，插件会尝试从对话中提取**结构化数据**（如用户的昵称、家乡、爱好、待办事项等），并像填表一样更新到数据库中。这让 Bot 能记住用户的具体属性。

3.  **长期上限 (融合机制)**:
    *   每日摘要如果无限积累，Token 也会爆炸。
    *   系统设有**每日摘要阈值**和**最大保留天数**。超过限制的旧摘要，会被 LLM 再次“压缩”，多条变一条，形成高度概括的“长期记忆”。

---

## ⚙️ 详细配置指南

您可以在 `data/plugins/astrbot_memory_manager/config.json` 中修改配置，或在 AstrBot WebUI 的插件配置页进行设置。

### 基础参数 (必看)

| 配置项 | 建议值 (群聊/私聊) | 详细说明 |
| :--- | :--- | :--- |
| `enable_auto_summary` | `true` | **总开关**。开启后才会自动执行总结任务。 |
| `trigger_threshold` (m) | `50` / `20` | **触发总结的消息量**。设太大容易丢失细节，设太小费 Token。群聊建议 50-80，私聊建议 20-30。 |
| `reserve_count` (n) | `10` / `5` | **总结时保留的最近消息数**。确保 Bot 在总结时还能"看"到最近几句对话，保持上下文连贯。建议为 `trigger_threshold` 的 10%-20%。 |
| `injection_count` | `3` / `5` | **注入摘要数量**。Bot 回复时，参考最近多少条摘要。设多了 Token 消耗大，设少了记不住事。 |
| `inject_all_summaries` | `false` | **注入所有摘要**。如果开启，会忽略 `injection_count`，尝试注入所有短期摘要（直到填满 `max_injection_tokens`）。适合对记忆完整性要求极高的场景，但**非常消耗 Token**。 |

### 进阶参数 (调优)

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `max_buffer_size` | `1000` | **缓冲区熔断保护**。如果 LLM 挂了导致无法总结，消息积压到这个数时会强制丢弃旧消息，防止内存溢出。 |
| `max_injection_tokens` | `2000` | **记忆注入上限**。为了防止记忆把 Prompt 撑爆，限制注入内容的最大 Token 数（约等于字符数）。 |
| `archive_time` | `"03:00"` | **每日整理时间**。建议设在夜深人静时。 |
| `max_retention_days` | `30` | **短期记忆保鲜期**。超过 30 天的每日摘要会被强制压缩成“长期记忆”并归档。 |
| `daily_summary_threshold` | `20` | **摘要密度控制**。如果一天内生成的摘要超过 20 条，触发合并压缩，防止碎片化太严重。 |

### LLM 相关 (可选)

默认情况下，插件复用 AstrBot 的 LLM 配置。如果您想用更便宜的模型（如 gpt-3.5-turbo-0125 或国产模型）专门处理记忆总结任务，可以单独配置：

| 配置项 | 说明 |
| :--- | :--- |
| `llm_api_base` | 自定义 API 地址 (例如 `https://api.deepseek.com/v1`) |
| `llm_api_key` | 自定义 API Key |
| `llm_model_name` | 自定义模型名称 (例如 `deepseek-chat`) |

---

## 🧩 与 AstrBot 原生功能的配合建议

**Q: AstrBot 本身就有上下文记忆，还需要开这个插件吗？**

**A: 强烈建议同时开启，两者互补。**

*   **AstrBot 原生上下文 (Working Memory)**:
    *   **作用**: 负责维护最近几轮的**精确**对话记录（User: A, Bot: B, User: C...）。
    *   **局限**: 长度有限（受限于 LLM 的 Context Window），聊久了最早的消息就被截断（遗忘）了。
    *   **建议**: 保持开启。这是 Bot 进行流畅对话的基础。

*   **本插件 (Long-term Memory)**:
    *   **作用**: 负责维护**长期的、高度概括的**背景知识和摘要。它注入的是 System Prompt，告诉 Bot "这个人过去发生过什么"、"他的偏好是什么"。
    *   **优势**: 即使是 30 天前聊过的内容，通过摘要归档也能被 Bot 感知到。

**最佳配置组合**:
1.  **AstrBot**: 保持默认上下文设置 (例如保留最近 10-20 轮)。
2.  **Memory Manager**:
    *   `trigger_threshold`: 50 (积累一定量再总结)
    *   `reserve_count`: 10 (总结时避开 AstrBot 正在处理的最近 10 条，避免逻辑冲突)
    *   `enable_structured_extraction`: true (记住用户画像)

---

## 💻 指令手册

所有指令前缀为 `/memory`。

### 基础操作
*   `/memory view`: 查看当前会话最近生成的记忆摘要。
*   `/memory view_long`: 查看归档的长期记忆。
*   `/memory stats`: 查看当前的内存、缓冲区统计数据。

### 重要事项管理 (高优先级注入)
Bot 会优先遵循这里的内容，适合存如"群规"、"用户称呼"、"绝对禁忌"等。
*   `/memory important add <内容>`: 添加一条重要事项。
*   `/memory important view`: 查看列表。
*   `/memory important del <序号>`: 删除。

### 结构化信息 (填表)
通常由 LLM 自动提取，也可以手动干预。
*   `/memory view_form`: 查看已提取的结构化数据。
*   `/memory fill_form <key> <value>`: 手动修正或添加数据（例如 `/memory fill_form 称呼 老板`）。

### 调试与维护
*   `/memory force_summary`: 强制立即执行一次总结（不等待阈值）。
*   `/memory clear_buffer`: 清空当前积压的消息（慎用，会导致丢失未总结的对话）。
*   `/memory export`: 导出当前会话的所有记忆数据为 JSON（便于备份或迁移）。
*   `/mem_show_t2i`: 专门查看与"文生图"相关的历史记忆。

---

## 📂 数据存储

插件数据存储在 `data/plugins/astrbot_memory_manager/memories/` 目录下。
*   每个会话（群聊/私聊）对应一个 `.json` 文件。
*   文件名为处理过的 Session ID（冒号被替换为 `_COLON_`）。
*   支持热迁移：直接备份或替换 json 文件即可。

---

**Author**: [木有知](https://github.com/muyouzhi6)
**Repo**: [https://github.com/muyouzhi6/astrbot_memory_manager](https://github.com/muyouzhi6/astrbot_memory_manager)

