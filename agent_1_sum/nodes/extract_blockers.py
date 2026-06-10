from agent_1_sum.chains import extract_blockers_chain
from common.logging import logger


def extract_blockers_node(
    transcript: str
) -> str:

    logger.info(
        "Extracting blockers"
    )

    response = extract_blockers_chain(
        transcript
    )

    return response.content