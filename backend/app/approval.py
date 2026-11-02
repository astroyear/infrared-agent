"""Human review uses a separate checkpoint thread for each durable proposal."""
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

class Review(TypedDict, total=False):
    operation_id: str
    decision: str | None

def build_review(saver):
    def review(state):
        decision = interrupt({"operation_id": state["operation_id"]})
        return {"decision": decision}

    graph = StateGraph(Review)
    graph.add_node("review", review)
    graph.add_edge(START, "review")
    graph.add_edge("review", END)
    return graph.compile(checkpointer=saver)

def config(op_id):
    return {"configurable": {"thread_id": "approval:" + op_id}}

def pause(graph, op_id):
    cfg = config(op_id)
    if not graph.get_state(cfg).values:
        graph.invoke({"operation_id": op_id, "decision": None}, cfg)

def decision(graph, op_id, action):
    cfg = config(op_id)
    saved = graph.get_state(cfg)
    if not saved.next and saved.values.get("decision") == action:
        return action
    if not saved.next:
        # Rebuild the checkpoint after a local restart or a rejected stale proposal.
        graph.invoke({"operation_id": op_id, "decision": None}, cfg)
    return graph.invoke(Command(resume=action), cfg)["decision"]
