import asyncio
import operator
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import Send, Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


load_dotenv(override=True)

MAX_ROUNDS = 3

# 并发探针：记录有多少个研究员同时运行
_active = 0
_peak = 0


# ---------- 运行时配置 ----------

class RunContext(TypedDict, total=False):
    supervisor_model: str
    researcher_model: str
    compression_model: str
    final_report_model: str
    temperature: float
    max_tokens: int


def default_model(env_name: str) -> str:
    """优先读角色专用环境变量；没设置时回退到 DEEPSEARCH_MODEL。"""
    return os.getenv(env_name) or os.getenv("DEEPSEARCH_MODEL", "")


class Configuration(BaseModel):
    supervisor_model: str = Field(
        default_factory=lambda: default_model("SUPERVISOR_MODEL"),
        description="主管拆题时使用的模型",
    )
    researcher_model: str = Field(
        default_factory=lambda: default_model("RESEARCHER_MODEL"),
        description="研究员搜索时使用的模型",
    )
    compression_model: str = Field(
        default_factory=lambda: default_model("COMPRESSION_MODEL"),
        description="压缩研究结果时使用的模型",
    )
    final_report_model: str = Field(
        default_factory=lambda: default_model("FINAL_REPORT_MODEL"),
        description="撰写最终报告时使用的模型",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, gt=0)

    @classmethod
    def from_context(
        cls,
        context: RunContext | None = None,
    ) -> "Configuration":
        # 本次调用传入的值覆盖默认值和环境变量
        return cls(**(context or {}))


def make_model(model_name: str, settings: Configuration) -> ChatAnthropic:
    """按角色配置创建一个 Anthropic 聊天模型。"""
    if not model_name:
        raise ValueError(
            "没有配置模型名。请在 .env 设置 DEEPSEARCH_MODEL，"
            "或设置对应的角色模型变量。"
        )

    return ChatAnthropic(
        model=model_name,
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )


def text_of(msg) -> str:
    """Anthropic 的 content 可能是块列表，统一取文本。"""
    content = msg.content
    if isinstance(content, str):
        return content

    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


# ---------- 工具 ----------

@tool
def search(query: str) -> str:
    """搜索 LangGraph、LangChain 和断点续跑相关资料。"""
    db = {
        "langgraph": (
            "LangGraph 由 LangChain 团队开发，用图结构编排有状态的 LLM 应用。"
        ),
        "langchain": (
            "LangChain 提供模型、工具和其他 LLM 应用开发组件。"
        ),
        "关系": (
            "LangGraph 专注于有状态、多步骤的流程编排；"
            "LangChain 提供模型和工具等基础组件。"
        ),
        "checkpoint": (
            "checkpointer 保存图的状态，支持断点续跑。"
        ),
    }

    hits = [
        value
        for keyword, value in db.items()
        if keyword in query.lower() or keyword in query
    ]
    return "\n".join(hits) if hits else f"没有找到关于「{query}」的资料。"


class ResearchComplete(BaseModel):
    """研究已完成，信息充足，可以开始撰写最终报告。"""


# ---------- 研究员子图 ----------

class ResearcherState(TypedDict):
    messages: Annotated[list, operator.add]
    topic: str
    rounds: int
    findings: Annotated[list, operator.add]


researcher_tools = [search, tool(ResearchComplete)]
researcher_tools_by_name = {
    one_tool.name: one_tool for one_tool in researcher_tools
}


def researcher_agent(
    state: ResearcherState,
    runtime: Runtime[RunContext],
) -> dict:
    global _active, _peak

    settings = Configuration.from_context(runtime.context)

    if state.get("rounds", 0) == 0:
        _active += 1
        _peak = max(_peak, _active)
        print(
            f"    [研究员] {state['topic'][:16]} 进场"
            f" | 同时在场={_active}"
            f" | 模型={settings.researcher_model}"
        )

    model = make_model(settings.researcher_model, settings)

    messages = [
        SystemMessage(
            content=(
                f"你是研究员，只负责这个子问题：【{state['topic']}】。"
                "需要资料时调用 search；信息足够后调用 ResearchComplete。"
            )
        )
    ] + state["messages"]

    response = model.bind_tools(researcher_tools).invoke(messages)

    print(
        f"    [研究员] 轮次={state.get('rounds', 0) + 1}"
        f" | 工具调用={[call['name'] for call in response.tool_calls]}"
    )

    return {
        "messages": [response],
        "rounds": state.get("rounds", 0) + 1,
    }


def researcher_call_tools(state: ResearcherState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        result = researcher_tools_by_name[tool_call["name"]].invoke(
            tool_call["args"]
        )
        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": tool_messages}


def researcher_should_continue(state: ResearcherState) -> str:
    last_message = state["messages"][-1]

    if state.get("rounds", 0) >= MAX_ROUNDS:
        return "compress"

    if not last_message.tool_calls:
        return "compress"

    if any(
        call["name"] == "ResearchComplete"
        for call in last_message.tool_calls
    ):
        return "compress"

    return "researcher_call_tools"


def compress(
    state: ResearcherState,
    runtime: Runtime[RunContext],
) -> dict:
    global _active

    settings = Configuration.from_context(runtime.context)
    model = make_model(settings.compression_model, settings)

    materials = "\n\n".join(
        message.content
        for message in state["messages"]
        if isinstance(message, ToolMessage)
    )

    response = model.invoke(
        [
            SystemMessage(
                content="根据资料，用 2-3 句话回答指定子问题，只输出答案。"
            ),
            HumanMessage(
                content=(
                    f"【子问题】{state['topic']}\n\n"
                    f"【资料】\n{materials or '(无)'}"
                )
            ),
        ]
    )

    _active -= 1
    return {"findings": [text_of(response)]}


researcher_builder = StateGraph(
    ResearcherState,
    context_schema=RunContext,
)
researcher_builder.add_node("researcher_agent", researcher_agent)
researcher_builder.add_node(
    "researcher_call_tools",
    researcher_call_tools,
)
researcher_builder.add_node("compress", compress)
researcher_builder.add_edge(START, "researcher_agent")
researcher_builder.add_conditional_edges(
    "researcher_agent",
    researcher_should_continue,
)
researcher_builder.add_edge(
    "researcher_call_tools",
    "researcher_agent",
)
researcher_builder.add_edge("compress", END)

researcher_graph = researcher_builder.compile()


# ---------- 主图：主管、并行研究员、汇总 ----------

class State(TypedDict):
    question: str
    topics: list
    findings: Annotated[list, operator.add]
    report: str


class Plan(BaseModel):
    topics: list[str] = Field(
        description="拆成 2-4 个可以独立研究的子问题"
    )


def supervisor(
    state: State,
    runtime: Runtime[RunContext],
) -> Command:
    settings = Configuration.from_context(runtime.context)
    model = make_model(settings.supervisor_model, settings)

    plan = model.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content="你是研究主管。把用户问题拆成 2-4 个可独立研究的子问题。"
            ),
            HumanMessage(content=state["question"]),
        ]
    )

    if plan is None:
        print("  [主管] 拆题失败，降级为单路研究")
        topics = [state["question"]]
    else:
        topics = plan.topics

    print(
        f"  [主管] 使用模型={settings.supervisor_model}"
        f" | 拆成 {len(topics)} 路"
    )

    return Command(
        goto=[
            Send(
                "researcher",
                {
                    "messages": [HumanMessage(content=topic)],
                    "topic": topic,
                    "rounds": 0,
                },
            )
            for topic in topics
        ],
        update={"topics": topics},
    )


def synthesize(
    state: State,
    runtime: Runtime[RunContext],
) -> dict:
    settings = Configuration.from_context(runtime.context)
    model = make_model(settings.final_report_model, settings)

    print(f"  [汇总] 使用模型={settings.final_report_model}")

    response = model.invoke(
        [
            SystemMessage(
                content="根据各子问题的结论，写一份完整、连贯的回答。"
            ),
            HumanMessage(
                content=(
                    f"【原问题】{state['question']}\n\n"
                    "【各子问题结论】\n"
                    + "\n\n".join(state["findings"])
                )
            ),
        ]
    )

    return {"report": text_of(response)}


builder = StateGraph(State, context_schema=RunContext)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher_graph)
builder.add_node("synthesize", synthesize)
builder.add_edge(START, "supervisor")
builder.add_edge("researcher", "synthesize")
builder.add_edge("synthesize", END)

app = builder.compile()


async def main():
    global _active, _peak
    _active = 0
    _peak = 0

    question = "LangGraph 是什么？它和 LangChain 什么关系？"

    # 留空时，每个角色从 .env 读取自己的模型名；
    # 没有角色专用变量时，回退到 DEEPSEARCH_MODEL。
    #
    # 也可以只覆盖本次运行的部分角色：
    # runtime_context: RunContext = {
    #     "researcher_model": "你的低成本模型名",
    #     "final_report_model": "你的高质量模型名",
    # }
    runtime_context: RunContext = {}

    result = await app.ainvoke(
        {
            "question": question,
            "findings": [],
        },
        config={"max_concurrency": 4},
        context=runtime_context,
    )

    print(f"\n峰值并发研究员数={_peak}")
    print("\n最终报告：\n", result["report"])


if __name__ == "__main__":
    asyncio.run(main())
