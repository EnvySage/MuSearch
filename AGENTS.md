# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 这是什么

**不是一个代码仓库,是一个学习项目。** 用户(2027 届软件工程本科,求职 AI 应用开发)正在从零重建 `langchain-ai/open_deep_research`,成品命名 **deepsearch**。

- **目的**:把 LangGraph 学透,做出第 4 个求职作品
- **定位**:作品集为主,顺带能用——冲突时优先「面试能讲」,但不做纯玩具
- **策略**:重建 → 加深 → 分化。先把架构学扎实,再逐步长成自己的东西

**交流一律用中文。**

## 协作规矩(最重要,优先于默认行为)

1. **不要直接写进他的文件,但要在对话里给出完整代码。**
   - ✅ **正确做法**:把完整代码贴在回复里 → 他读懂 → 自己敲进文件 → 运行
   - ❌ **不要**:用 Write/Edit 替他改项目源文件
   - **理由**:他不熟悉 LangChain 语法和依赖,**只给骨架会让他卡死**。看代码是学语法,敲代码是记语法,两件事都要做。
   - **例外**:纯环境配置(`.env`、`.gitignore`、依赖安装)可以直接写文件。
2. **遇到新名词,用大白话解释并举例。** 先给类比或例子,再给定义,不要默认他懂。
3. **对他的观点先对立,后协作——但只提一次。**
   - 他提出看法时,先认真找一个**真实**的反驳点,逼他想清楚最坏情况。
   - **但说完就走**:他说"我知道""我懂了这个",**立刻接受,不再重复同一论点**。
   - **按他的节奏。** 他要往前走就往前走。同一个提醒说到第三遍是干扰,不是帮助。
   - 想留的提醒记进 `docs/IMPROVEMENTS.md` 或当天日志,**不占用他的时间反复拦他**。
   - 要真实的反驳,不是抬杠凑数。找不到真实问题就直说。
4. **每天记学习日志,作为他写博客的素材。** 见 `docs/journal/`。
   - 记的是**素材**,不是文章:卡在哪、推翻了什么想法、能写成什么角度
   - **每天记录,但不必每天成文**——硬凑会生产垃圾,然后他就不写了
   - 重点是**冲突和解法**,不是"今天做了什么"的流水账
   - **绝不写入密钥、内网地址等敏感信息**——日志可能变成公开博客

## 目录结构

| 路径 | 是什么 |
|---|---|
| `open_deep_research/` | **上游参考仓库**(已归档只读)。当教材读,**不要修改** |
| `.venv/` | 最小学习环境:Python 3.11 + langgraph 1.x |
| `.env.example` | 环境变量模板,复制成 `.env` 填自己的 key |
| `docs/` | 学习文档(见下表) |
| `main.py` | 练习入口,当前为空 |

`docs/` 里的文档:

| 文件 | 用途 | 更新频率 |
|---|---|---|
| `LEARNING.md` | 学习路径——8 级概念阶梯,读 `open_deep_research` 的顺序 | 基本不动 |
| `PROGRESS.md` | 进度日志 + 每级验收标准。**每次学完记一行** | 每次 |
| `NOTES.md` | 术语表——新名词用大白话 + 例子记一遍 | 常更新 |
| `IMPROVEMENTS.md` | 改进清单——重建中冒出的"我会换个做法"全记这 | 常更新 |
| `journal/YYYY-MM-DD.md` | **每日学习素材**,供他写博客。见规矩 4 | 每次 |

`open_deep_research/` 有它自己的 `AGENTS.md`,那个是上游的,与本文件无关。

## 重建阶梯(Roadmap)

每一级都能跑、都只教一个概念。**做完一级再进下一级。**

| 版本 | 加什么 | 学到什么 | 原项目对应 |
|---|---|---|---|
| v0.1 | `START → 搜索 → 总结 → END` | `StateGraph` / state / node / edge / `compile` | 最小骨架 |
| v0.2 | researcher ⇄ tools 循环 + 出口条件 | ReAct、`@tool`、`ToolMessage`、**怎么保证它停** | `deep_researcher.py:365-509` |
| v0.3 | supervisor 拆题 + 并行 researcher | 子图、`asyncio.gather`、`Command` 路由 | `353-363`、`295-305` |
| v0.4 | `Configuration` 类 + 多模型角色 | `config_schema`、运行时配置 | `configuration.py` |
| v0.5 | checkpointer + `interrupt` | 持久化、断点续跑、人机协同 | `legacy/graph.py:142-192` |

**v0.2 是分水岭**——做完它才真正理解 agent 的本质。

**注意排序问题**:`v0.4`(配置)原计划排第四,但**调试前四级时只能用贵模型**。若成本敏感,应提前到 v0.2 之后。

## 常用命令

```bash
# 激活环境
.venv\Scripts\activate

# 装包(用 uv,别用全局 pip)
uv pip install --python .venv <package>

# 跑练习
.venv\Scripts\python.exe main.py
```

## 已知事实

- `open_deep_research` 要求 `langgraph>=0.5.4`,实测 **1.2.15 完全兼容**——它用到的 `StateGraph` / `Command` / `MessagesState` / `interrupt` 在 1.x 全部可用。
- 上游仓库默认配置**一次完整运行 $20~100**(gpt-4.1 × 5 并行研究员 × 6 轮 × 10 次工具调用)。学习时把 `max_concurrent_research_units` / `max_researcher_iterations` / `max_react_tool_calls` 调到 1,降到几分钱。
- 上游的 `utils.py`(925 行)里**约 700 行是集成胶水**(Tavily 客户端、MCP 接入、20 个搜索引擎适配),**不是学习重点**。真正的教学核心是 `deep_researcher.py`(718)+ `state.py`(95)+ `configuration.py`(251)。

## 待补充

- [ ] `.env`——复制 `.env.example` 后填入自己的 key(用户说本地已有,具体 provider 待确认)
