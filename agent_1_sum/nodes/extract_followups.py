from agent_1_sum.chains import extract_followups_chain
from common.logging import logger


def extract_followups_node(
    transcript: str
) -> str:

    logger.info(
        "Extracting follow-ups"
    )

    response = extract_followups_chain(
        transcript
    )

    return response.content