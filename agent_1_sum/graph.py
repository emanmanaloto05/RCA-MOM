from langgraph.graph import END
from langgraph.graph import StateGraph

from agent_1_sum.models import MOMState
from agent_1_sum.nodes.extract_blockers import extract_blockers_node
from agent_1_sum.nodes.extract_followups import extract_followups_node
from agent_1_sum.nodes.extract_owners import extract_owners_node
from agent_1_sum.nodes.extract_tasks import extract_tasks_node
from agent_1_sum.nodes.generate_mom import generate_mom_node
from agent_1_sum.nodes.summarize import summarize_node
from agent_1_sum.nodes.transcribe import transcribe_audio


def transcribe_graph_node(state: MOMState):
    state["transcript"] = transcribe_audio(
        state["audio_path"]
    )

    return state


def summarize_graph_node(state: MOMState):
    state["summary"] = summarize_node(
        state["transcript"]
    )

    return state


def owners_graph_node(state: MOMState):
    state["owners"] = extract_owners_node(
        state["transcript"]
    )

    return state


def tasks_graph_node(state: MOMState):
    state["tasks"] = extract_tasks_node(
        state["transcript"]
    )

    return state


def blockers_graph_node(state: MOMState):
    state["blockers"] = extract_blockers_node(
        state["transcript"]
    )

    return state


def followups_graph_node(state: MOMState):
    state["followups"] = extract_followups_node(
        state["transcript"]
    )

    return state


def mom_graph_node(state: MOMState):
    state["mom"] = generate_mom_node(
        summary=state["summary"],
        owners=state["owners"],
        tasks=state["tasks"],
        blockers=state["blockers"],
        followups=state["followups"]
    )

    return state


def build_mom_graph():
    graph = StateGraph(MOMState)

    graph.add_node("transcribe", transcribe_graph_node)
    graph.add_node("summarize", summarize_graph_node)
    graph.add_node("owners", owners_graph_node)
    graph.add_node("tasks", tasks_graph_node)
    graph.add_node("blockers", blockers_graph_node)
    graph.add_node("followups", followups_graph_node)
    graph.add_node("generate_mom", mom_graph_node)

    graph.set_entry_point("transcribe")

    graph.add_edge("transcribe", "summarize")
    graph.add_edge("summarize", "owners")
    graph.add_edge("owners", "tasks")
    graph.add_edge("tasks", "blockers")
    graph.add_edge("blockers", "followups")
    graph.add_edge("followups", "generate_mom")
    graph.add_edge("generate_mom", END)

    return graph.compile()


mom_graph = build_mom_graph()