import os
import operator
from typing import Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from pydantic import BaseModel

load_dotenv()

# model = ChatOpenAI(
#     model=os.getenv("DEEPSEARCH_MODEL"),
#     base_url=os.getenv("OPENAI_BASE_URL"),
#     api_key=os.getenv("OPENAI_API_KEY"),
# )
MAX_ROUNDS = 10
model = ChatAnthropic(
    model=os.getenv("DEEPSEARCH_MODEL"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4096,        # Anthropic 必须给输出上限,不给会用默认值
    temperature= 0
)

def text_of(msg) -> str:
    """Anthropic 的 content 可能是块列表,统一取出文本部分。"""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(
        b.get("text", "") for b in c
        if isinstance(b, dict) and b.get("type") == "text"
    )


# ---------- 工具(假的,不用联网) ----------
@tool
def search(query: str) -> str:
    """在需要查找 LangGraph / LangChain / 断点续跑相关资料时调用。

    已经获得足够信息回答用户问题后,不要再调用本工具。
    每次调用只查一个具体关键词。
    """
    db = {
        "langgraph": "LangGraph 由 LangChain 团队开发,2024 年发布,用图结构编排 LLM 应用。",
        "langchain": "LangChain 是 LLM 应用开发框架,2022 年开源。",
        "checkpoint": "checkpointer 把图的状态存下来,支持断点续跑。",
    }
    for k, v in db.items():
        if k in query.lower():
            return v
    return f"没有找到关于「{query}」的资料。"

class ResearchComplete(BaseModel):
    """研究已完成,信息充足,可以开始撰写最终报告。"""
# ---------- 状态 ----------
class State(TypedDict):
    messages: Annotated[list, operator.add]   # ★ 这一行是 v0.2 的核心
    rounds: int


# ---------- 节点 ----------
tools = [search, tool(ResearchComplete)]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


def agent(state: State) -> dict:
    msgs = [SystemMessage(content="你是研究助手。需要资料时调用 search 工具。请你一定要在回答的结尾加上喵~")] + state["messages"]
    r = model_with_tools.invoke(msgs)
    print(f"  [agent] 第 {state.get('rounds', 0) + 1} 轮 | 想调: {[t['name'] for t in r.tool_calls]}")
    return {"messages": [r], "rounds": state.get("rounds", 0) + 1}


def call_tools(state: State) -> dict:
    last = state["messages"][-1]
    out = []
    for tc in last.tool_calls:
        res = tools_by_name[tc["name"]].invoke(tc["args"])
        print(f"  [tool ] {tc['args']} -> {res[:28]}...")
        out.append(ToolMessage(content=res, tool_call_id=tc["id"]))
    return {"messages": out}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if state.get("rounds", 0) >= MAX_ROUNDS:
        print("  [路由] 达到轮数上限 -> write_report")
        return "write_report"
    if not last.tool_calls:
        print("  [路由] 模型没再调工具 -> write_report")
        return "write_report"
    if any(tc["name"] == "ResearchComplete" for tc in last.tool_calls):
        print("  [路由] 模型主动说完成了 -> write_report")     # ← 新增
        return "write_report"
    return "call_tools"

def write_report(state: State) -> dict:
    # 只取"材料"(工具返回的正文),丢掉"动作"(AI 的工具调用记录)
    materials = "\n\n".join(
        m.content for m in state["messages"] if isinstance(m, ToolMessage)
    )
    msgs = [
        SystemMessage(content=(
            "根据下面收集到的资料,写一份完整的回答。"
            "直接输出正文,不要调用任何工具,不要输出任何工具调用格式。结尾回答加上喵"
        )),
        HumanMessage(content=materials),
    ]
    r = model.invoke(msgs)
    return {"messages": [r]}


# ---------- 组装 ----------
g = StateGraph(State)
g.add_node("agent", agent)
g.add_node("call_tools", call_tools)
g.add_node("write_report", write_report)
g.add_edge(START, "agent")
g.add_conditional_edges("agent", should_continue)   # ★ 动态路由
g.add_edge("call_tools", "agent")                   # ★ 回头箭头 = 循环
g.add_edge("write_report", END)

app = g.compile()

out = app.invoke({
    "messages": [HumanMessage(content="LangGraph 是什么?它和 LangChain 什么关系?")],
    "rounds": 0,
})

print("\n最终回答:", text_of(out["messages"][-1]))
