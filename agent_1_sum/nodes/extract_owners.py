from agent_1_sum.chains import extract_owners_chain
from common.logging import logger


def extract_owners_node(
    transcript: str
) -> str:

    logger.info(
        "Extracting owners"
    )

    response = extract_owners_chain(
        transcript
    )

    return response.content