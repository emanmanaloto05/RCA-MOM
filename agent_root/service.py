"""
RCA Service — orchestrates the full RCA generation pipeline via LangGraph,
then renders HTML and converts to PDF using Playwright.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agent_root.graph import GraphBuildError, RCAGraphState, get_rca_graph
from agent_root.models import (
    ApprovalStatus,
    DeveloperIssueData,
    QualityGateData,
    RCAInputModel,
    RCAOutputModel,
)

logger = logging.getLogger("rca_generator.service")


class RCAServiceError(RuntimeError):
    """Raised when the RCA generation pipeline fails."""


@dataclass
class RCASectionData:
    issue_number: int
    issue_title: str
    issue_description: str
    product: str
    module: str
    core_function: str

    cause: str = "Not specified."
    impact_analysis: str = "Not specified."
    solution: str = "Not specified."
    preventive_action: str = "Not specified."
    owner_review: str = "Not specified."

    quality_gate_pass: str = "N/A"
    smoke_test_pass: str = "N/A"
    reopen_count: int = 0
    fc_failed: int = 0
    qa_status: str = "N/A"
    pic_qa: str = "N/A"

    attachments: list[str] = field(
        default_factory=lambda: cast(list[str], [])
    )


def _extract_section(markdown: str, headings: list[str]) -> str:
    for heading in headings:
        pattern = rf"{re.escape(heading)}\s*(.*?)(?=\n##\s|\Z)"
        match = re.search(pattern, markdown, flags=re.DOTALL)

        if match:
            body = match.group(1).strip()
            if body and not body.startswith("#"):
                return body

    return "Not specified."


def extract_markdown_sections(markdown_rca: str) -> dict[str, str]:
    return {
        "cause": _extract_section(
            markdown_rca,
            [
                "## 2. Root Cause",
                "## Root Cause",
                "## 2. Cause",
                "## Cause",
            ],
        ),
        "impact_analysis": _extract_section(
            markdown_rca,
            [
                "## 3. Impact Analysis",
                "## Impact Analysis",
            ],
        ),
        "solution": _extract_section(
            markdown_rca,
            [
                "## 6. Corrective Action",
                "## Corrective Action",
                "## 6. Solution",
                "## Solution",
            ],
        ),
        "preventive_action": _extract_section(
            markdown_rca,
            [
                "## 7. Preventive Action",
                "## Preventive Action",
            ],
        ),
        "owner_review": _extract_section(
            markdown_rca,
            [
                "## 8. Owner Review",
                "## Owner Review",
            ],
        ),
    }


def _bool_to_pass_fail(value: bool | None) -> str:
    if value is True:
        return "Passed"

    if value is False:
        return "Failed"

    return "N/A"


def _resolve_consultant(
    developer: DeveloperIssueData | None,
    quality: QualityGateData | None,
) -> str:
    if developer and developer.pic_dev:
        return developer.pic_dev

    if quality and quality.pic_qa:
        return quality.pic_qa

    return ""


def cleanup_old_outputs(days: int = 30) -> None:
    output_dir = Path("outputs")

    if not output_dir.exists():
        return

    cutoff = datetime.now() - timedelta(days=days)

    for file_path in output_dir.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {".html", ".pdf"}:
            continue

        modified_at = datetime.fromtimestamp(file_path.stat().st_mtime)

        if modified_at >= cutoff:
            continue

        try:
            file_path.unlink()
            logger.info("Deleted old output file | path=%s", file_path)

        except Exception as exc:
            logger.warning(
                "Failed to delete old output file | path=%s | error=%s",
                file_path,
                exc,
            )


def _generate_pdf_from_html(html_path: Path, pdf_path: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]

        logger.info(
            "Playwright: converting HTML to PDF | html=%s | pdf=%s",
            html_path,
            pdf_path,
        )

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()

            page.goto(html_path.resolve().as_uri())
            page.wait_for_load_state("networkidle")

            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={
                    "top": "16mm",
                    "bottom": "20mm",
                    "left": "18mm",
                    "right": "18mm",
                },
            )

            browser.close()

        logger.info("Playwright: PDF written | pdf=%s", pdf_path)

    except ImportError as exc:
        raise RCAServiceError(
            "Playwright is not installed. "
            "Run: pip install playwright && playwright install chromium"
        ) from exc

    except Exception as exc:
        logger.exception(
            "Playwright PDF conversion failed | html=%s | error=%s",
            html_path,
            exc,
        )
        raise RCAServiceError(
            f"PDF generation failed for {html_path.name}: {exc}"
        ) from exc


def save_audit_log(
    issue_id: str,
    client: str,
    generated_by: str,
    pdf_file_path: str | None,
    status: str,
) -> None:
    audit_dir = Path("audit_logs")
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_data: dict[str, str | None] = {
        "issue_id": issue_id,
        "client": client,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": generated_by,
        "pdf_file_path": pdf_file_path,
        "status": status,
    }

    audit_file = audit_dir / f"{issue_id}.json"

    audit_file.write_text(
        json.dumps(
            audit_data,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info("Audit log written | path=%s", audit_file)


class RCAService:
    TEMPLATE_DIR: Path = Path("templates")
    TEMPLATE_NAME: str = "rca_template.html"
    OUTPUT_DIR: Path = Path("outputs")
    LOGO_PATH: str = "static/images/direc_logo.png"
    CSS_PATH: str = "static/rca.css"

    @classmethod
    async def generate_rca(cls, rca_input: RCAInputModel) -> RCAOutputModel:
        cleanup_old_outputs()

        issue_id = rca_input.task_monitoring_data.issue_logs_id

        if not issue_id.strip():
            raise RCAServiceError("Issue ID is required for RCA generation.")

        logger.info("RCA generation started | issue_id=%s", issue_id)

        initial_state: RCAGraphState = {
            "rca_input": rca_input,
        }

        try:
            graph: Any = get_rca_graph()

            final_state_raw: Any = await asyncio.to_thread(
                graph.invoke,
                initial_state,
            )

            if not isinstance(final_state_raw, dict):
                raise RCAServiceError(
                    f"Invalid graph output type for issue {issue_id}: "
                    f"{type(final_state_raw).__name__}"
                )

            final_state = cast(dict[str, Any], final_state_raw)

        except GraphBuildError as exc:
            logger.exception(
                "Graph build failed | issue_id=%s | error=%s",
                issue_id,
                exc,
            )
            raise RCAServiceError(
                f"RCA workflow could not be initialized for issue {issue_id}."
            ) from exc

        except RCAServiceError:
            raise

        except Exception as exc:
            logger.exception(
                "Graph invocation failed | issue_id=%s | error=%s",
                issue_id,
                exc,
            )
            raise RCAServiceError(
                f"RCA workflow failed for issue {issue_id}."
            ) from exc

        if not final_state.get("review_passed", False):
            error_detail = (
                final_state.get("generation_error")
                or final_state.get("review_notes")
                or "Unknown RCA review failure."
            )

            logger.error(
                "RCA review failed | issue_id=%s | detail=%s",
                issue_id,
                error_detail,
            )

            raise RCAServiceError(
                f"RCA generation did not pass review for issue {issue_id}: "
                f"{error_detail}"
            )

        rca_output = final_state.get("rca_output")

        if not isinstance(rca_output, RCAOutputModel):
            logger.error(
                "Invalid or missing RCA output | issue_id=%s | type=%s",
                issue_id,
                type(rca_output).__name__,
            )
            raise RCAServiceError(
                f"RCA output is missing or invalid for issue {issue_id}."
            )

        if not rca_output.markdown_rca.strip():
            raise RCAServiceError(
                f"Generated RCA markdown is empty for issue {issue_id}."
            )

        html_file_path = cls.generate_html_file(
            rca_input=rca_input,
            rca_output=rca_output,
        )

        pdf_file_path = await cls.generate_pdf_file(
            html_path=Path(html_file_path),
        )

        rca_output.pdf_file_path = f"/api/rca/{issue_id}/download"
        rca_output.approval_status = ApprovalStatus.DRAFT

        save_audit_log(
            issue_id=issue_id,
            client=rca_input.task_monitoring_data.client,
            generated_by="RCA Generator",
            pdf_file_path=pdf_file_path,
            status=rca_output.approval_status.value,
        )

        logger.info(
            "RCA generation completed | issue_id=%s | output_chars=%d | pdf=%s",
            issue_id,
            len(rca_output.markdown_rca),
            pdf_file_path,
        )

        return rca_output

    @classmethod
    def generate_html_file(
        cls,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> str:
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        issue_id = rca_input.task_monitoring_data.issue_logs_id
        output_path = cls.OUTPUT_DIR / f"{issue_id}_rca.html"

        html_content = cls.render_html(
            rca_input=rca_input,
            rca_output=rca_output,
        )

        output_path.write_text(html_content, encoding="utf-8")

        logger.info("HTML written | path=%s", output_path)

        return str(output_path)

    @classmethod
    async def generate_pdf_file(cls, html_path: Path) -> str:
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        pdf_path = html_path.with_suffix(".pdf")

        await asyncio.to_thread(
            _generate_pdf_from_html,
            html_path,
            pdf_path,
        )

        return str(pdf_path)

    @classmethod
    def render_html(
        cls,
        rca_input: RCAInputModel,
        rca_output: RCAOutputModel,
    ) -> str:
        env = Environment(
            loader=FileSystemLoader(str(cls.TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )

        template = env.get_template(cls.TEMPLATE_NAME)

        task = rca_input.task_monitoring_data
        quality = rca_input.quality_gate_data
        github_pr = rca_input.github_pr
        developer = rca_input.developer_issue_data

        sections = extract_markdown_sections(rca_output.markdown_rca)

        qg_pass = _bool_to_pass_fail(
            quality.quality_gate_first_pass if quality else None
        )

        smoke_pass = _bool_to_pass_fail(
            quality.smoke_test_first_pass if quality else None
        )

        qa_status = (
            quality.qa_status.value
            if quality and quality.qa_status
            else "N/A"
        )

        pic_qa = (
            quality.pic_qa
            if quality and quality.pic_qa
            else "N/A"
        )

        reopen = quality.reopen_count if quality else 0
        fc_failed = quality.fc_failed_testing if quality else 0

        owner_review = (
            github_pr.owner_review
            if github_pr and github_pr.owner_review
            else sections.get("owner_review", "Pending owner review.")
        )

        rca_section = RCASectionData(
            issue_number=1,
            issue_title=task.title,
            issue_description=task.issue_description,
            product=task.product,
            module=task.module,
            core_function=task.core_function,
            cause=sections.get("cause", "Not specified."),
            impact_analysis=sections.get(
                "impact_analysis",
                "Not specified.",
            ),
            solution=sections.get("solution", "Not specified."),
            preventive_action=sections.get(
                "preventive_action",
                "Not specified.",
            ),
            owner_review=owner_review,
            quality_gate_pass=qg_pass,
            smoke_test_pass=smoke_pass,
            reopen_count=reopen,
            fc_failed=fc_failed,
            qa_status=qa_status,
            pic_qa=pic_qa,
            attachments=[],
        )

        return template.render(
            task=task,
            all_issue_titles=[task.title],
            generated_date=datetime.now().strftime("%B %d, %Y"),
            assigned_consultant=_resolve_consultant(
                developer,
                quality,
            ),
            approval_status=rca_output.approval_status.value,
            rca_sections=[rca_section],
            logo_path=cls.LOGO_PATH,
            css_path=cls.CSS_PATH,
        )