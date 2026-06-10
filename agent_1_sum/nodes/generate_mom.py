from common.logging import logger


def generate_mom_node(
    summary: str,
    owners: str,
    tasks: str,
    blockers: str,
    followups: str
) -> str:

    logger.info(
        "Generating MOM document"
    )

    mom = f"""
# Minutes of Meeting

## Summary
{summary}

## Owners
{owners}

## Tasks
{tasks}

## Blockers
{blockers}

## Follow-ups
{followups}
"""

    return mom