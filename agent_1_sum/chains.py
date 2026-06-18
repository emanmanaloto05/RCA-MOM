from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

from common.logging import logger
from config.providers import (
    openai_llm,
    gemini_summary_llm,
    gemini_mom_llm
)


PROMPTS_PATH = (
    Path(__file__).resolve().parent
    / "prompts.yaml"
)


def load_prompts() -> dict:
    with open(
        PROMPTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


def build_prompt(
    prompt_name: str
) -> ChatPromptTemplate:
    prompts = load_prompts()

    prompt_config = prompts[prompt_name]

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt_config["system"]
            ),
            (
                "user",
                prompt_config["user"]
            )
        ]
    )


def get_backup_llm(task_name: str):
    if task_name == "summary generation":
        return gemini_summary_llm

    if task_name == "MOM generation":
        return gemini_mom_llm

    if task_name == "supervisor final review":
        return gemini_mom_llm

    return gemini_summary_llm


def invoke_with_fallback(
    prompt: ChatPromptTemplate,
    values: dict,
    task_name: str
) -> str:
    try:
        logger.info(
            f"Using OpenAI primary model for {task_name}"
        )

        chain = prompt | openai_llm

        response = chain.invoke(
            values
        )

        return extract_response_text(
            response
        )

    except Exception as error:
        logger.warning(
            f"OpenAI failed for {task_name}. "
            f"Switching to Gemini fallback. Error: {error}"
        )

        backup_llm = get_backup_llm(task_name)

        backup_chain = prompt | backup_llm

        response = backup_chain.invoke(
            values
        )

        return extract_response_text(
            response
        )


def extract_response_text(response) -> str:
    content = getattr(
        response,
        "content",
        response
    )

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        extracted_parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    extracted_parts.append(
                        item.get("text", "")
                    )
                elif "text" in item:
                    extracted_parts.append(
                        item.get("text", "")
                    )
            else:
                extracted_parts.append(
                    str(item)
                )

        return "\n".join(
            part for part in extracted_parts if part
        )

    return str(content)


def generate_summary_chain(
    transcript: str
):
    prompt = build_prompt(
        "summarizer_agent"
    )

    return invoke_with_fallback(
        prompt=prompt,
        values={
            "transcript": transcript
        },
        task_name="summary generation"
    )


def generate_mom_chain(
    summary: str,
    transcript: str
):
    prompt = build_prompt(
        "mom_generator_agent"
    )

    return invoke_with_fallback(
        prompt=prompt,
        values={
            "summary": summary,
            "transcript": transcript
        },
        task_name="MOM generation"
    )


def supervisor_final_review_chain(
    transcript: str,
    summary: str,
    mom: str
):
    prompt = build_prompt(
        "supervisor_final_review"
    )

    return invoke_with_fallback(
        prompt=prompt,
        values={
            "transcript": transcript,
            "summary": summary,
            "mom": mom
        },
        task_name="supervisor final review"
    )