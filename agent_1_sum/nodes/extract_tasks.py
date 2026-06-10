from agent_1_sum.chains import extract_tasks_chain
from common.logging import logger


def extract_tasks_node(
    transcript: str
) -> str:

    logger.info(
        "Extracting tasks"
    )

    response = extract_tasks_chain(
        transcript
    )

    return response.content