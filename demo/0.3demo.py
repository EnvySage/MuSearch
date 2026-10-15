import asyncio, os, operator, time
from typing import Annotated
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command          # ★ 新增:Send 和 Command
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

load_dotenv(override=True)

model = ChatAnthropic(
    model=os.getenv("DEEPSEARCH_MODEL"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4096,
    temperature=0,
)
MAX_ROUNDS = 3

# ---- 并发探针:数"同时在场"的研究员 ----
_active = 0
_peak = 0


def text_of(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")


@tool
def search(query: str) -> str:
    """在需要查找 LangGraph / LangChain / 断点续跑相关资料时调用。
    已经获得足够信息回答该子问题后,不要再调用本工具。
    每次调用只查一个具体关键词。"""
    db = {
        "langgraph": "LangGraph 由 LangChain 团队开发,2024 年发布,用图结构编排 LLM 应用,核心抽象是 StateGraph/Node/Edge。",
        "langchain": "LangChain 是 LLM 应用开发框架,2022 年开源,提供模型/工具/链(LCEL)等基础组件。",
        "关系": "LangGraph 是 LangChain 生态的一部分,专注有状态、可循环的多步编排;LangChain 提供上层组件与集成。",
        "checkpoint": "checkpointer 把图的状态存下来,支持断点续跑。",
    }
    hits = [v for k, v in db.items() if k in query.lower() or k in query]
    return "\n".join(hits) if hits else f"没有找到关于「{query}」的资料。"


class ResearchComplete(BaseModel):
    """研究已完成,信息充足,可以开始撰写最终报告。"""


# ===================== 子图:单个研究员 =====================
class ResearcherState(TypedDict):
    messages: Annotated[list, operator.add]
    topic: str
    rounds: int
    findings: Annotated[list, operator.add]     # ★ 改动1:str → list(跟父图对齐)


r_tools_list = [search, tool(ResearchComplete)]
r_by_name = {t.name: t for t in r_tools_list}
r_model = model.bind_tools(r_tools_list)


def r_agent(state: ResearcherState) -> dict:
    global _active, _peak
    if state.get("rounds", 0) == 0:             # 只在"第一次进场"计数
        _active += 1
        _peak = max(_peak, _active)
        print(f"    [研究员] {state['topic'][:16]} 进场 | 同时在场 = {_active}")
    msgs = [SystemMessage(content=(
        f"你是研究员,只负责这一个子问题:【{state['topic']}】。"
        "需要资料时调用 search;信息足够后调用 ResearchComplete。"
    ))] + state["messages"]
    r = r_model.invoke(msgs)
    print(f"    [研究员 agent] {state['topic'][:10]} 轮 {state.get('rounds', 0) + 1} | 想调 {[t['name'] for t in r.tool_calls]}")
    return {"messages": [r], "rounds": state.get("rounds", 0) + 1}


def r_call_tools(state: ResearcherState) -> dict:
    last = state["messages"][-1]
    out = []
    for tc in last.tool_calls:
        res = r_by_name[tc["name"]].invoke(tc["args"])
        out.append(ToolMessage(content=res, tool_call_id=tc["id"]))
    return {"messages": out}


def r_should_continue(state: ResearcherState) -> str:
    last = state["messages"][-1]
    if state.get("rounds", 0) >= MAX_ROUNDS:
        return "compress"
    if not last.tool_calls:
        return "compress"
    if any(tc["name"] == "ResearchComplete" for tc in last.tool_calls):
        return "compress"
    return "r_call_tools"


def compress(state: ResearcherState) -> dict:
    global _active
    materials = "\n\n".join(m.content for m in state["messages"] if isinstance(m, ToolMessage))
    r = model.invoke([
        SystemMessage(content="根据下面的资料,用 2-3 句话回答指定子问题。只输出答案本身。"),
        HumanMessage(content=f"【子问题】{state['topic']}\n\n【资料】\n{materials or '(无)'}"),
    ])
    _active -= 1
    return {"findings": [text_of(r)]}            # ★ 改动2:包成 list


rb = StateGraph(ResearcherState)
rb.add_node("r_agent", r_agent)
rb.add_node("r_call_tools", r_call_tools)
rb.add_node("compress", compress)
rb.add_edge(START, "r_agent")
rb.add_conditional_edges("r_agent", r_should_continue)
rb.add_edge("r_call_tools", "r_agent")
rb.add_edge("compress", END)
researcher = rb.compile()                        # 编译出来,等下当 Send 目标


# ===================== 主图:主管 + Send 扇出 =====================
class State(TypedDict):
    question: str
    topics: list
    findings: Annotated[list, operator.add]      # ★ 改动3:加 reducer(并行写必需)
    report: str


class Plan(BaseModel):
    topics: list[str] = Field(description="拆解出的 2-4 个可各自独立检索的子问题")


def supervisor(state: State) -> Command:
    plan = model.with_structured_output(Plan).invoke([
        SystemMessage(content="你是研究主管。把用户的问题拆成 2-4 个可以各自独立检索的子问题。"),
        HumanMessage(content=state["question"]),
    ])
    if plan is None:
        print("  [主管] 拆题失败,降级为单路")
        topics = [state["question"]]
    else:
        topics = plan.topics
    print(f"  [主管] 拆成 {len(topics)} 路,Send 一次性扇出:")
    for i, t in enumerate(topics, 1):
        print(f"         {i}. {t}")
    # ★ 改动4:返回 Command —— 路由 + 状态一起给,不用再写条件边
    return Command(
        goto=[
            Send("researcher", {
                "messages": [HumanMessage(content=t)],
                "topic": t,
                "rounds": 0,
            })
            for t in topics
        ],
        update={"topics": topics},
    )


def synthesize(state: State) -> dict:
    r = model.invoke([
        SystemMessage(content="根据下面各子问题的结论,写一份完整、连贯的回答。结尾加上喵~"),
        HumanMessage(content=f"【原问题】{state['question']}\n\n【各子问题结论】\n" + "\n\n".join(state["findings"])),
    ])
    return {"report": text_of(r)}


g = StateGraph(State)
g.add_node("supervisor", supervisor)
g.add_node("researcher", researcher)
g.add_node("synthesize", synthesize)
g.add_edge(START, "supervisor")
g.add_edge("researcher", "synthesize")
g.add_edge("synthesize", END)
app = g.compile()


async def main():
    global _active, _peak
    q = "LangGraph 是什么?它和 LangChain 什么关系?"

    # # ---- 第 1 遍:不加限制 ----
    # _active = _peak = 0
    # t0 = time.time()
    # out = await app.ainvoke({"question": q, "findings": []})
    # print(f"\n[第1遍 无限制] 总耗时 {time.time()-t0:.1f}s,峰值并发 = {_peak}")
    # print("最终报告:\n", out["report"])

    # ---- 第 2 遍:打开全局闸门,把这 4 行前面的 # 去掉 ----
    _active = _peak = 0
    t0 = time.time()
    out = await app.ainvoke({"question": q, "findings": []}, {"max_concurrency": 1})
    print(f"\n[第2遍 限流=1] 总耗时 {time.time()-t0:.1f}s,峰值并发 = {_peak}")
    print("最终报告:\n", out["report"])


if __name__ == "__main__":
    asyncio.run(main())
