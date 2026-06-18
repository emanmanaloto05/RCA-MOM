from datetime import datetime
from pathlib import Path
import time
from jinja2 import Template
from langgraph.graph import END
from langgraph.graph import StateGraph
from openai import OpenAI
from google import genai

from config.settings import settings

gemini_client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)

openai_client = OpenAI(
    api_key=settings.OPENAI_API_KEY
)

def transcribe_with_openai(
    audio_path: str
) -> str:
    logger.info(
        "Using OpenAI primary transcriber"
    )

    with open(
        audio_path,
        "rb"
    ) as audio_file:
        response = openai_client.audio.transcriptions.create(
            model=settings.OPENAI_TRANSCRIBE_MODEL,
            file=audio_file
        )

    return response.text.strip()


def transcribe_with_gemini(
    audio_path: str
) -> str:
    logger.info(
        "Using Gemini fallback transcriber"
    )

    uploaded_file = gemini_client.files.upload(
        file=audio_path
    )

    retry_delays = [3, 10]

    last_error = None

    for attempt in range(1, 4):
        try:
            logger.info(
                f"Gemini transcription attempt {attempt}/3"
            )

            response = gemini_client.models.generate_content(
                model=settings.GEMINI_TRANSCRIBE_MODEL,
                contents=[
                    (
                        "Transcribe this meeting audio accurately. "
                        "Return only the transcript text. "
                        "Do not summarize. Do not add comments."
                    ),
                    uploaded_file,
                ],
            )

            return response.text.strip()

        except Exception as error:
            last_error = error

            if attempt == 3:
                break

            wait_time = retry_delays[attempt - 1]

            logger.warning(
                f"Gemini transcription failed on attempt {attempt}. "
                f"Retrying in {wait_time} seconds. Error: {error}"
            )

            time.sleep(wait_time)

    raise RuntimeError(
        f"Gemini transcription failed after 3 attempts: {last_error}"
    )

from agent_1_sum.services.pdf_generator import generate_pdf_from_html
from agent_1_sum.chains import (
    generate_mom_chain,
    generate_summary_chain,
    supervisor_final_review_chain
)
from agent_1_sum.models import MOMState
from agent_1_sum.services.docx_generator import generate_docx
from common.exceptions import TranscriptionException
from common.logging import logger


SECTION_HEADERS = [
    "SUBJECT:",
    "ATTENDEES:",
    "AGENDA:",
    "EXECUTIVE SUMMARY:",
    "DISCUSSION POINTS:",
    "DECISIONS MADE:",
    "ACTION ITEMS:",
    "OWNERS:",
    "RISKS / ISSUES / BLOCKERS:",
    "FOLLOW-UP ACTIONS:",
    "MEETING OUTCOME:",
    "FINAL REVIEW STATUS:",
    "QUALITY SCORE:",
    "FINAL APPROVAL:",
]


def clean_ai_text(text) -> str:
    if not text:
        return "Not Specified"

    if isinstance(text, list):
        text = "\n".join(
            str(item)
            for item in text
        )

    if not isinstance(text, str):
        text = str(text)

    cleaned_text = (
        text.replace("\\n", "\n")
        .replace("\\r", "")
        .replace("**", "")
        .replace("##", "")
        .replace("`", "")
        .replace("* ", "")
        .replace("- ", "")
        .replace("{'type': 'text', 'text':", "")
        .replace("'extras':", "")
        .strip()
    )

    if "extras" in cleaned_text:
        cleaned_text = cleaned_text.split(
            "extras"
        )[0].strip()

    cleaned_text = cleaned_text.strip("'").strip('"').strip()

    return cleaned_text or "Not Specified"


def extract_section(
    content: str,
    start_label: str
) -> str:
    normalized_content = clean_ai_text(
        content
    )

    start_index = normalized_content.find(
        start_label
    )

    if start_index == -1:
        return "Not Specified"

    start_index += len(
        start_label
    )

    end_index = len(
        normalized_content
    )

    for header in SECTION_HEADERS:
        if header == start_label:
            continue

        header_index = normalized_content.find(
            header,
            start_index
        )

        if header_index != -1:
            end_index = min(
                end_index,
                header_index
            )

    section = normalized_content[
        start_index:end_index
    ].strip()

    return clean_ai_text(
        section
    )


def format_lines_as_html(text: str) -> str:
    cleaned_text = clean_ai_text(
        text
    )

    if cleaned_text == "Not Specified":
        return "Not Specified"

    lines = [
        line.strip()
        for line in cleaned_text.splitlines()
        if line.strip()
    ]

    if not lines:
        return "Not Specified"

    if len(lines) == 1:
        return lines[0]

    html_lines = [
        f"<div>{line}</div>"
        for line in lines
    ]

    return "\n".join(
        html_lines
    )


def format_action_items_as_html(text: str) -> str:
    cleaned_text = clean_ai_text(
        text
    )

    if cleaned_text == "Not Specified":
        return "Not Specified"

    blocks = cleaned_text.split(
        "Task:"
    )

    formatted_items = []

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip()
        ]

        if not lines:
            continue

        task = lines[0]
        details = lines[1:]

        detail_html = ""

        for detail in details:
            if ":" in detail:
                label, value = detail.split(
                    ":",
                    1
                )

                detail_html += (
                    "<br>"
                    f"<span class='highlight'>{label.strip()}:</span> "
                    f"{value.strip()}"
                )
            else:
                detail_html += (
                    "<br>"
                    f"{detail}"
                )

        formatted_items.append(
            "<div class='task-item'>"
            f"<span class='highlight'>Task:</span> {task}"
            f"{detail_html}"
            "</div>"
        )

    if not formatted_items:
        return cleaned_text

    return "\n".join(
        formatted_items
    )


def transcribe_node(state: MOMState):
    try:
        audio_path = state["audio_path"]

        logger.info(
            f"Starting transcription for: {audio_path}"
        )

        try:
            transcript = transcribe_with_openai(
                audio_path
            )

        except Exception as error:
            logger.warning(
                f"OpenAI transcription failed. "
                f"Switching to Gemini fallback. Error: {error}"
            )

            transcript = transcribe_with_gemini(
                audio_path
            )

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


def generate_summary_node(state: MOMState):
    logger.info(
        "Generating structured meeting summary"
    )

    response = generate_summary_chain(
        state["transcript"]
    )

    structured_summary = clean_ai_text(
        response
    )

    output_dir = Path(
        "agent_1_sum/output/summaries"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_name = (
        f"Meeting_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    summary_path = output_dir / file_name

    summary_path.write_text(
        structured_summary,
        encoding="utf-8"
    )

    state["structured_summary"] = structured_summary
    state["summary_path"] = str(
        summary_path
    )

    return state


def analyze_meeting_node(state: MOMState):
    logger.info(
        "Extracting MOM details from transcript"
    )

    response = generate_mom_chain(
        summary=state["structured_summary"],
        transcript=state["transcript"]
    )

    content = clean_ai_text(
        response
    )

    state["subject"] = extract_section(
        content,
        "SUBJECT:"
    )

    state["attendees"] = extract_section(
        content,
        "ATTENDEES:"
    )

    state["agenda"] = extract_section(
        content,
        "AGENDA:"
    )

    state["summary"] = extract_section(
        content,
        "EXECUTIVE SUMMARY:"
    )

    state["decisions_made"] = extract_section(
        content,
        "DECISIONS MADE:"
    )

    state["tasks"] = extract_section(
        content,
        "ACTION ITEMS:"
    )

    state["owners"] = extract_section(
        content,
        "OWNERS:"
    )

    state["blockers"] = extract_section(
        content,
        "RISKS / ISSUES / BLOCKERS:"
    )

    state["followups"] = extract_section(
        content,
        "FOLLOW-UP ACTIONS:"
    )

    state["meeting_outcome"] = extract_section(
        content,
        "MEETING OUTCOME:"
    )

    return state


def supervisor_review_node(state: MOMState):
    logger.info(
        "Running supervisor final review"
    )

    mom_draft = (
        f"SUBJECT:\n{state['subject']}\n\n"
        f"ATTENDEES:\n{state['attendees']}\n\n"
        f"AGENDA:\n{state['agenda']}\n\n"
        f"EXECUTIVE SUMMARY:\n{state['summary']}\n\n"
        f"DECISIONS MADE:\n{state['decisions_made']}\n\n"
        f"ACTION ITEMS:\n{state['tasks']}\n\n"
        f"OWNERS:\n{state['owners']}\n\n"
        f"RISKS / ISSUES / BLOCKERS:\n{state['blockers']}\n\n"
        f"FOLLOW-UP ACTIONS:\n{state['followups']}\n\n"
        f"MEETING OUTCOME:\n{state['meeting_outcome']}"
    )

    review = supervisor_final_review_chain(
        transcript=state["transcript"],
        summary=state["structured_summary"],
        mom=mom_draft
    )

    review_text = clean_ai_text(
        review
    )

    state["supervisor_review"] = review_text

    if "FINAL REVIEW STATUS:" in review_text:
        review_status = extract_section(
            review_text,
            "FINAL REVIEW STATUS:"
        )

        if "REJECTED" in review_status.upper():
            logger.warning(
                "Supervisor rejected MOM draft. Continuing generation with review notes."
            )

    return state


def generate_mom_node(state: MOMState):
    logger.info(
        "Generating MOM HTML and DOCX documents"
    )

    generated_date = datetime.now().strftime(
        "%B %d, %Y"
    )

    generated_time = datetime.now().strftime(
        "%I:%M %p"
    )

    template_path = Path(
        "agent_1_sum/templates/mom_template.html"
    )

    template_content = template_path.read_text(
        encoding="utf-8"
    )

    template = Template(
        template_content
    )

    generated_date_full = datetime.now().strftime(
        "%B %d, %Y %I:%M %p"
    )

    document_id = (
        f"MOM-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    html_content = template.render(
        company_name="Direc Business Technologies Inc.",

        generated_date=generated_date_full,
        document_id=document_id,

        subject=clean_ai_text(
            state["subject"]
        ),
        date=generated_date,
        time=generated_time,
        attendees=format_lines_as_html(
            state["attendees"]
        ),
        agenda=format_lines_as_html(
            state["agenda"]
        ),
        summary=format_lines_as_html(
            state["summary"]
        ),
        decisions_made=format_lines_as_html(
            state["decisions_made"]
        ),
        tasks=format_action_items_as_html(
            state["tasks"]
        ),
        owners=format_lines_as_html(
            state["owners"]
        ),
        blockers=format_lines_as_html(
            state["blockers"]
        ),
        followups=format_lines_as_html(
            state["followups"]
        ),
        meeting_outcome=format_lines_as_html(
            state["meeting_outcome"]
        )
    )

    html_output_dir = Path(
        "agent_1_sum/output/html"
    )

    html_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    html_file_name = (
        f"Meeting_Notes_{file_timestamp}.html"
    )

    html_path = html_output_dir / html_file_name

    html_path.write_text(
        html_content,
        encoding="utf-8"
    )

    logger.info(
        f"MOM HTML saved to: {html_path}"
    )

    pdf_path = generate_pdf_from_html(
    html_path=str(html_path),
    file_name=f"Meeting_Notes_{file_timestamp}.pdf"
)

    state["pdf_path"] = pdf_path

    state["docx_path"] = ""

    if state.get("generate_docx", False):

        docx_path = generate_docx(
            company_name="Direc Business Technologies Inc.",
            subject=clean_ai_text(
                state["subject"]
            ),
            date=generated_date,
            time=generated_time,
            attendees=clean_ai_text(
                state["attendees"]
            ),
            agenda=clean_ai_text(
                state["agenda"]
            ),
            summary=clean_ai_text(
                state["summary"]
            ),
            decisions_made=clean_ai_text(
                state["decisions_made"]
            ),
            tasks=clean_ai_text(
                state["tasks"]
            ),
            owners=clean_ai_text(
                state["owners"]
            ),
            blockers=clean_ai_text(
                state["blockers"]
            ),
            followups=clean_ai_text(
                state["followups"]
            ),
            meeting_outcome=clean_ai_text(
                state["meeting_outcome"]
            )
        )

        logger.info(
            f"MOM DOCX saved to: {docx_path}"
        )

        state["docx_path"] = docx_path

    state["html_path"] = str(
        html_path
    )

    state["mom"] = html_content

    return state


def build_mom_graph():
    graph = StateGraph(
        MOMState
    )

    graph.add_node(
        "transcribe",
        transcribe_node
    )

    graph.add_node(
        "generate_summary",
        generate_summary_node
    )

    graph.add_node(
        "analyze_meeting",
        analyze_meeting_node
    )

    graph.add_node(
        "supervisor_review",
        supervisor_review_node
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
        "generate_summary"
    )

    graph.add_edge(
        "generate_summary",
        "analyze_meeting"
    )

    graph.add_edge(
        "analyze_meeting",
        "supervisor_review"
    )

    graph.add_edge(
        "supervisor_review",
        "generate_mom"
    )

    graph.add_edge(
        "generate_mom",
        END
    )

    return graph.compile()


mom_graph = build_mom_graph()