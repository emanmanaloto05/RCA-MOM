from datetime import datetime

from common.logging import logger


def audit_log(action: str):
    logger.info(
        f"[AUDIT] {datetime.now()} | {action}"
    )