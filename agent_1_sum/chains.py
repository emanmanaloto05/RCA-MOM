from langchain_core.prompts import PromptTemplate

from config.providers import llm

def summarize_chain(
    transcript: str
):
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""
        Summarize the following meeting transcript.

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
    
def extract_owners_chain(
    transcript: str
):
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""
        Extract all owners and responsible persons.

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
    
def extract_tasks_chain(
    transcript: str
):
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""
        Extract all action items and tasks.

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
    
def extract_blockers_chain(
    transcript: str
):
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""
        Extract blockers, issues, risks,
        and concerns.

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
    
def extract_followups_chain(
    transcript: str
):
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template="""
        Extract follow-up actions
        and next steps.

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