# 用 open_deep_research 学 LangGraph

环境已建好：`.venv`（Python 3.11 + langgraph 1.2.15），仓库在 `open_deep_research/`。
已确认仓库用到的 API（`StateGraph` / `Command` / `MessagesState` / `interrupt`）在 1.x 全部可用。

**不要按文件顺序读。** 按下面这个概念阶梯读，每一级只解决一个问题。

---

## 第 0 级：先自己写一个 15 行的图（别读仓库）

在看任何代码前，先手写一遍。LangGraph 的心智模型只有 4 个东西：

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    n: Annotated[list, operator.add]   # reducer：见第 1 级

def a(s): return {"n": ["a"]}
def b(s): return {"n": ["b"]}

g = StateGraph(S)
g.add_node("a", a); g.add_node("b", b)
g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", END)
print(g.compile().invoke({"n": []}))    # {'n': ['a', 'b']}
```

看懂这 15 行，后面全是它的变体。**先写再读**，否则你会以为 LangGraph 很复杂。

---

## 第 1 级：state.py —— 状态怎么合并（95 行，但最重要）

`src/open_deep_research/state.py`

LangGraph 最难的不是节点，是**状态合并语义**。这个文件讲透了：

| 看什么 | 行 | 学到 |
|---|---|---|
| `Annotated[list, operator.add]` | 68 | 默认是**覆盖**，加了 reducer 才**追加** |
| `override_reducer` | 55-60 | 自定义 reducer：值形如 `{"type":"override","value":x}` 就整体替换，否则追加 |
| `MessagesState` 继承 | 62, 65 | `MessagesState` = 自带 messages 键 + add reducer 的 TypedDict |
| 每个子图独立的 State | 74, 83, 92 | Supervisor / Researcher / Output 三套状态，**互不污染** |

关键点：`AgentState` 和 `SupervisorState` **不是同一个 state**。子图有自己的状态空间，主图通过 `Command(update=...)` 往里灌。这是新手最容易搞混的地方。

自定义 reducer 的用法在第 163-174 行：
```python
"supervisor_messages": {"type": "override", "value": [SystemMessage(...), HumanMessage(...)]}
```
**为什么要这个？** 因为 messages 默认是追加，但初始化 supervisor 时需要**清空重写**。

---

## 第 2 级：最小的完整图（12 行）

`src/open_deep_research/deep_researcher.py:351-363`

```python
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()
```

注意 **`config_schema=Configuration`** —— 这是生产代码的标志，见第 6 级。
注意这里 `compile()` 出来的东西是**完整的图对象**，下一级会把它当节点用。

---

## 第 3 级：ReAct 循环（最经典的 agent 结构）

`deep_researcher.py:365-509`（节点 `researcher` ⇄ `researcher_tools`）

```
START → researcher ──→ researcher_tools ──┬──→ researcher   (继续)
              ↑                           └──→ compress_research → END
              └───────────────────────────┘
```

这是全宇宙所有 agent 的骨架。**重点看出口条件**：

| 出口 | 行 | 条件 |
|---|---|---|
| 早退 | 463-464 | 没有 tool call（模型直接给了答案） |
| 迭代上限 | 492 | `tool_call_iterations >= max_react_tool_calls` |
| 主动结束 | 493-496 | 模型调了 `ResearchComplete` 工具 |

**这才是 agent 工程的核心**：怎么保证它一定会停。所有 agent 的 bug 一半出在这。

---

## 第 4 级：`Command` 路由（新写法）

`deep_researcher.py:217-223`、`256-262`、`418-424`

每个节点不返回值，而是返回：

```python
return Command(goto="supervisor_tools", update={"supervisor_messages": [response]})
```

把「**状态更新**」和「**下一步去哪**」合并成一次返回。

**对比**：老代码用 `add_conditional_edges`（边函数决定去哪），状态和路由是分开的。
新写法把路由逻辑收进节点内部，更容易读。

看 `supervisor_tools`（225-349）怎么用**一个函数**实现「继续循环 / 结束」两种路由：
靠 `if` 判断，返回 `goto="supervisor"` 或 `goto=END`。不需要任何条件边。

---

## 第 5 级：子图组合 + 并行

**子图编译**：`deep_researcher.py:589-605` → `researcher_subgraph`

**两种把子图塞进主图的方式**，这个仓库两种都用了：

```python
# 方式 A：作为节点（主图用）
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)   # 710 行

# 方式 B：手动调用（并行 fan-out 用）
research_tasks = [researcher_subgraph.ainvoke({...}, config) for tool_call in calls]
tool_results = await asyncio.gather(*research_tasks)                           # 295-305 行
```

**并行 fan-out 的两种做法，仓库也两种都有**：

| 做法 | 位置 | 特点 |
|---|---|---|
| `asyncio.gather` 手动并发 | 现版本 `deep_researcher.py:305` | 简单直接，你自己控并发数 |
| `Send()` API | legacy `graph.py:181, 466`、`multi_agent.py:306` | LangGraph 官方机制，由框架调度 |

**这是个重要的判断题**：官方推荐 `Send`，但这个仓库的现版本选择了 `asyncio`。
原因值得你自己想——提示：`Send` 的并发数你控制不了，而 API 成本要靠 `max_concurrent_research_units` 卡死（291 行）。

---

## 第 6 级：配置系统（从 demo 到生产的门槛）

`src/open_deep_research/configuration.py` + `deep_researcher.py:74`

```python
configurable = Configuration.from_runnable_config(config)
if not configurable.allow_clarification:
    ...
```

**这一层决定你的图是玩具还是能上线的。** 关键点：
- 所有可调参数进一个 Pydantic 类，`config_schema=Configuration` 注册
- `from_runnable_config` 先读环境变量，再读 `config["configurable"]`
- 于是同一个图，LangGraph Studio 里改配置就能换模型 / 换搜索源，**不用改代码**

`Configuration` 有 4 个模型角色（research / compression / summarization / final_report），每个有独立的 `max_tokens`。**多模型分工**是省钱的核心手段。

---

## 第 7 级：读老版本 —— 最好的教学材料

`src/legacy/graph.py` 和 `legacy/multi_agent.py` 是**同一个问题的上一代解法**。
对照着读，你能看到 LangGraph 这几年的设计演进：

| 能力 | 老写法（legacy） | 新写法（现版本） |
|---|---|---|
| 路由 | `add_conditional_edges`（499 行） | `Command(goto=...)` |
| 人工介入 | `interrupt()`（170-175 行） | 现版本删掉了 |
| 并行 | `Send()`（181, 466 行） | `asyncio.gather` |
| 分工 | 计划-执行，人工审批计划 | 主管-研究员，全自动 |

**`interrupt()` 那段一定要看**（`legacy/graph.py:142-192`）：这是 LangGraph 做人机协同的标准做法——
节点里调 `interrupt(问题)`，图会**暂停并把状态存进 checkpointer**，等人回复后从断点恢复。
只有配合持久化才有意义，这是 checkpoint 机制最直观的用例。

作者为此写了篇 "bitter lesson" 博客：**砍掉了计划-审批流程，让模型拿工具自己跑，效果反而更好**。
这个取舍比代码本身值钱。

---

## 便宜跑起来（重要）

默认配置一次完整运行 **$20~100**。因为默认是 `gpt-4.1` × 5 个并行研究员 × 6 轮迭代 × 每轮 10 次工具调用。

学习用不着这么跑。把规模压到最小：

`.env`：
```bash
# 用便宜模型（DeepSeek 极便宜；或 anthropic:claude-haiku-4-5）
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...          # 免费额度 1000 次/月
LANGSMITH_TRACING=false          # 先别开，省得看花眼
```

在 LangGraph Studio 的配置里把这三个调成 1：
- `max_concurrent_research_units` = 1
- `max_researcher_iterations` = 1
- `max_react_tool_calls` = 2

**一次运行从 $45 降到几分钱**，足够看清整条链路。

启动：
```bash
cd open_deep_research
uv run --project ../.venv langgraph dev --allow-blocking
```
> 注意：跑完整仓库需要它的全套依赖（`uv sync`，约 1GB）。只想读代码的话不用装。
> 上面的 `.venv` 是给**你自己写练习**用的最小环境。

---

## 练习阶梯（这才是真正学会的方式）

读完一级就动手改一次，不要一口气读完。

1. **改第 0 级的图**：加一个节点，体会 `Annotated` reducer 去掉后的差别（列表会怎样？）
2. **抄第 2 级**：把 supervisor 子图默写一遍，跑通
3. **抄第 3 级**：写一个只有 2 个工具的 ReAct 循环，**故意把出口条件删掉**，看它怎么死
4. **加一个工具**：仿照 `utils.py:220` 的 `think_tool`，写自己的工具（看它的 docstring 怎么写——模型靠这个决定何时调用）
5. **加人机协同**：仿照 `legacy/graph.py:170` 给第 3 级的图加 `interrupt()`
6. **写自己的配置类**：把第 3 级里写死的模型名换成 `Configuration`

做完 1-6，去看 `deep_researcher.py` 就再也没有看不懂的地方了。

---

## 一页速查

| 概念 | 位置 |
|---|---|
| State + reducer | `state.py:55-95` |
| 最小完整图 | `deep_researcher.py:351-363` |
| ReAct 循环 | `deep_researcher.py:365-509` |
| Command 路由 | `deep_researcher.py:217-223` |
| 子图当节点 | `deep_researcher.py:710` |
| 并行 fan-out | `deep_researcher.py:295-305` |
| Send API | `legacy/graph.py:181` / `multi_agent.py:306` |
| interrupt 人机协同 | `legacy/graph.py:142-192` |
| 旧式条件边 | `legacy/graph.py:499` |
| 自定义工具 | `utils.py:220` |
| 配置系统 | `configuration.py` |
| 结构化输出 | `deep_researcher.py:89-94` |
