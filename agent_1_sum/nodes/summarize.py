from agent_1_sum.chains import summarize_chain
from common.logging import logger


def summarize_node(
    transcript: str
) -> str:

    logger.info(
        "Generating meeting summary"
    )

    response = summarize_chain(
        transcript
    )

    return response.content