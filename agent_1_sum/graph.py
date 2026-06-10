from langgraph.graph import StateGraph
from langgraph.graph import END

from agent_1_sum.models import MOMState
from agent_1_sum.service import generate_mom_data


def generate_mom_node(
    state: MOMState
):
    transcript = state["transcript"]

    result = generate_mom_data(
        transcript
    )

    state["summary"] = result["summary"]
    state["owners"] = result["owners"]
    state["tasks"] = result["tasks"]
    state["blockers"] = result["blockers"]
    state["followups"] = result["followups"]

    return state


def build_mom_graph():
    graph = StateGraph(MOMState)

    graph.add_node(
        "generate_mom",
        generate_mom_node
    )

    graph.set_entry_point(
        "generate_mom"
    )

    graph.add_edge(
        "generate_mom",
        END
    )

    return graph.compile()


mom_graph = build_mom_graph()