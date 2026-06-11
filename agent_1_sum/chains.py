from langchain_core.prompts import PromptTemplate

from config.providers import llm


def summarize_chain(
    transcript: str
):
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""
        Summarize the complete meeting transcript clearly and professionally.

        Make sure no important decisions, concerns, tasks, or discussion points
        are missed.

        Transcript:
        {transcript}
        """
    )

    chain = prompt | llm

    return chain.invoke(
        {
            "transcript": transcript
        }
    )


def analyze_meeting_chain(
    summary: str
):
    prompt = PromptTemplate(
        input_variables=["summary"],
        template="""
        Analyze the meeting summary below.

        Extract the following sections only:

        OWNERS:
        - List all people responsible for tasks, decisions, or follow-ups.

        TASKS:
        - List all tasks, action items, deliverables, and mentioned work.

        BLOCKERS:
        - List all blockers, risks, concerns, issues, or pending problems.

        FOLLOWUPS:
        - List all follow-up actions, next steps, target dates, or pending items.

        Meeting Summary:
        {summary}
        """
    )

    chain = prompt | llm

    return chain.invoke(
        {
            "summary": summary
        }
    )