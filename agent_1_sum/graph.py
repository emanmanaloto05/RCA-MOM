from pathlib import Path

import whisper
from langgraph.graph import END
from langgraph.graph import StateGraph

from agent_1_sum.chains import (
    analyze_meeting_chain,
    summarize_chain
)
from agent_1_sum.models import MOMState
from common.exceptions import TranscriptionException
from common.logging import logger


def transcribe_node(state: MOMState):
    try:
        audio_path = state["audio_path"]

        logger.info(
            f"Starting transcription for: {audio_path}"
        )

        model = whisper.load_model("base")

        result = model.transcribe(
            audio_path
        )

        transcript = result.get(
            "text",
            ""
        ).strip()

        output_dir = Path(
            "agent_1_sum/output/transcripts"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        audio_name = Path(
            audio_path
        ).stem

        transcript_path = (
            output_dir
            / f"{audio_name}_transcript.txt"
        )

        transcript_path.write_text(
            transcript,
            encoding="utf-8"
        )

        logger.info(
            f"Transcript saved to: {transcript_path}"
        )

        state["transcript"] = transcript
        state["transcript_path"] = str(
            transcript_path
        )

        return state

    except Exception as error:
        logger.error(
            f"Transcription failed: {error}"
        )

        raise TranscriptionException(
            "Failed to transcribe audio file"
        ) from error


def summarize_node(state: MOMState):
    logger.info(
        "Generating meeting summary"
    )

    response = summarize_chain(
        state["transcript"]
    )

    state["summary"] = response.content

    return state


def analyze_meeting_node(state: MOMState):
    logger.info(
        "Analyzing meeting summary for owners, tasks, blockers, and follow-ups"
    )

    response = analyze_meeting_chain(
        state["summary"]
    )

    analysis = response.content

    state["owners"] = analysis
    state["tasks"] = analysis
    state["blockers"] = analysis
    state["followups"] = analysis

    return state


def generate_mom_node(state: MOMState):
    logger.info(
        "Generating MOM document"
    )

    state["mom"] = f"""
# Minutes of Meeting

## Transcript File
{state["transcript_path"]}

## Summary
{state["summary"]}

## Meeting Analysis
{state["owners"]}
"""

    return state


def build_mom_graph():
    graph = StateGraph(MOMState)

    graph.add_node(
        "transcribe",
        transcribe_node
    )

    graph.add_node(
        "summarize",
        summarize_node
    )

    graph.add_node(
        "analyze_meeting",
        analyze_meeting_node
    )

    graph.add_node(
        "generate_mom",
        generate_mom_node
    )

    graph.set_entry_point(
        "transcribe"
    )

    graph.add_edge(
        "transcribe",
        "summarize"
    )

    graph.add_edge(
        "summarize",
        "analyze_meeting"
    )

    graph.add_edge(
        "analyze_meeting",
        "generate_mom"
    )

    graph.add_edge(
        "generate_mom",
        END
    )

    return graph.compile()


mom_graph = build_mom_graph()