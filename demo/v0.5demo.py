import sys
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class ReviewState(TypedDict):
    draft: str
    approved: bool
    result: str


def human_review(state: ReviewState) -> dict:
    approved = interrupt({
        "question": "批准这份报告吗？",
        "draft": state["draft"],
    })
    return {"approved": approved}


def route_after_review(state: ReviewState) -> str:
    return "approved" if state["approved"] else "rejected"


def approved(state: ReviewState) -> dict:
    return {"result": f"已批准：{state['draft']}"}


def rejected(state: ReviewState) -> dict:
    return {"result": "已拒绝，需要修改。"}


builder = StateGraph(ReviewState)
builder.add_node("human_review", human_review)
builder.add_node("approved", approved)
builder.add_node("rejected", rejected)

builder.add_edge(START, "human_review")
builder.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "approved": "approved",
        "rejected": "rejected",
    },
)
builder.add_edge("approved", END)
builder.add_edge("rejected", END)


# 数据库文件会保存在 demo 目录下
with SqliteSaver.from_conn_string("demo/checkpoints.sqlite") as checkpointer:
    app = builder.compile(checkpointer=checkpointer)

    # 两次运行必须使用相同的 thread_id，才能找到同一条运行记录
    config = {"configurable": {"thread_id": "report-001"}}

    if len(sys.argv) != 2 or sys.argv[1] not in {"start", "resume"}:
        print("用法：")
        print("  python demo/v0.5demo.py start")
        print("  python demo/v0.5demo.py resume")

    elif sys.argv[1] == "start":
        paused = app.invoke(
            {
                "draft": "LangGraph 使用图结构编排有状态流程。",
                "approved": False,
                "result": "",
            },
            config,
        )

        interrupt_info = paused["__interrupt__"][0]
        print("图已暂停，等待审批：", interrupt_info.value)
        print("现在可以关闭程序；之后运行 resume 模式继续。")

    else:  # resume
        while True:
            answer = input("批准吗？（y/n）：").strip().lower()

            if answer in {"y", "yes"}:
                resume_value = True
                break
            elif answer in {"n", "no"}:
                resume_value = False
                break
            else:
                print("请输入 y 或 n。")

        finished = app.invoke(Command(resume=resume_value), config)
        print(finished["result"])
