from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
AttachmentDict = dict[str, str | bool | None]
RcaSection     = dict[str, Any]


def resolve_attachment(filename: str, static_dir: Path) -> AttachmentDict:
    """
    Given an attachment filename, return a dict with the original name
    and a resolved file URI (if the file exists inside static/images).
    Falls back to None for the uri if the file is not found.
    """
    images_dir: Path = static_dir / "images"
    file_path:  Path = images_dir / filename

    result: AttachmentDict = {
        "name":     filename,
        "uri":      file_path.resolve().as_uri() if file_path.exists() else None,
        "is_image": filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")
        ),
    }
    return result


def main() -> None:
    root_dir: Path = Path(__file__).resolve().parent

    templates_dir: Path = root_dir / "templates"
    static_dir:    Path = root_dir / "static"
    output_dir:    Path = root_dir / "outputs"

    template_file: Path = templates_dir / "rca_template.html"
    css_file:      Path = static_dir / "rca.css"
    logo_file:     Path = static_dir / "images" / "direc_logo.png"

    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")

    if not css_file.exists():
        raise FileNotFoundError(f"CSS not found: {css_file}")

    if not logo_file.exists():
        raise FileNotFoundError(f"Logo not found: {logo_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=True,
    )

    template = env.get_template("rca_template.html")

    # ---------------------------------------------------------------------------
    # RCA sections — attachments are resolved to file URIs so the template
    # can render <img> tags (or download links for non-images) directly.
    # ---------------------------------------------------------------------------
    rca_sections: list[RcaSection] = [
        {
            "issue_number": 1,

            "issue_title":
            "Attendance records not updating after biometric synchronization",

            "issue_description": """
The Human Resource department reported an issue where employee
attendance records were not being updated after the scheduled
biometric synchronization process.

Employees were able to successfully clock in and clock out using
the biometric devices. However, the attendance information was not
properly displayed inside the HRIS Attendance Management module.

The issue caused inconsistencies between the physical biometric logs
and the records stored in the system database.
""",

            "product": "People Navee HRIS",

            "module": "Attendance Management",

            "core_function": "Biometric Synchronization Service",


            "cause": """
<h3>Technical Root Cause Analysis</h3>

<p>
After conducting an investigation, the development team identified
that the failure occurred within the backend synchronization process
responsible for transferring biometric device logs into the HRIS
database.
</p>

<p>
The synchronization service encountered timeout errors whenever
a large volume of employee attendance records was processed.
The API connection was automatically terminated before the system
completed the transaction.
</p>

<p>
Several contributing factors were discovered:
</p>

<ul>
<li>
The biometric API timeout configuration was too short for large
attendance batches.
</li>

<li>
The system processed records sequentially instead of using optimized
batch processing.
</li>

<li>
There was no automatic retry mechanism when synchronization failed.
</li>

<li>
Error monitoring did not immediately notify administrators about
failed synchronization attempts.
</li>
</ul>

<p>
The issue was classified as a backend service reliability problem
affecting data synchronization.
</p>
""",


            "affected_module": """
<h3>Affected System Components</h3>

<p>
The issue affected multiple HRIS components that depend on accurate
attendance information.
</p>

<ul>
<li>Attendance Monitoring Dashboard</li>
<li>Employee Daily Time Records</li>
<li>Biometric Integration Service</li>
<li>Payroll Attendance Validation</li>
<li>Attendance Report Generation</li>
</ul>

<p>
Other HRIS modules remained operational and no employee profile
information was affected.
</p>
""",


            "impact_analysis": """
<h3>Business Impact Analysis</h3>

<p>
The synchronization failure affected daily HR operations because
attendance records required manual verification before processing.
</p>

<p>
Employees who successfully recorded their attendance through the
biometric device appeared as absent or incomplete in the system.
</p>

<p>
Potential operational impacts included:
</p>

<ul>
<li>Incorrect attendance monitoring results</li>
<li>Possible payroll computation discrepancies</li>
<li>Additional manual checking workload for HR personnel</li>
<li>Delay in attendance approval workflow</li>
<li>Reduced reliability of automated reporting</li>
</ul>
""",


            "quality_gate_pass": "Failed",

            "smoke_test_pass": "Passed",

            "reopen_count": 1,

            "fc_failed": "Yes",

            "qa_status": "Validated After Fix",

            "pic_qa": "Jane Arianne A. Gaela",


            "solution": """
<h3>Corrective Action Implemented</h3>

<p>
The development team modified the synchronization service to improve
performance, reliability, and failure handling.
</p>

<p>
Implemented solutions:
</p>

<ul>
<li>
Increased biometric API timeout limit from 30 seconds to 120 seconds.
</li>

<li>
Implemented retry mechanism using exponential backoff strategy.
</li>

<li>
Optimized attendance processing using database batch transactions.
</li>

<li>
Added detailed error logging for failed synchronization attempts.
</li>

<li>
Created validation checks before marking synchronization as complete.
</li>
</ul>

<p>
After deployment, the synchronization process was tested with
large attendance datasets and completed successfully.
</p>
""",


            "preventive_action": """
<h3>Preventive Action Plan</h3>

<p>
Preventive improvements were planned to avoid similar issues in
future releases.
</p>

<ul>
<li>
Implement automated synchronization monitoring alerts.
</li>

<li>
Create daily attendance reconciliation reports.
</li>

<li>
Add automated regression test cases for biometric integration.
</li>

<li>
Perform scheduled performance testing on high-volume transactions.
</li>

<li>
Improve API failure handling documentation.
</li>
</ul>
""",


            "owner_review": """
<h3>System Owner Review</h3>

<p>
The system owner reviewed the implemented corrective actions,
QA validation results, and testing evidence.
</p>

<p>
The solution successfully restored attendance synchronization
functionality and improved system reliability.
</p>

<p>
The issue was approved for closure after confirming that no
additional attendance inconsistencies were detected.
</p>
""",


            # -------------------------------------------------------------------
            # Attachments: filenames are resolved to file:// URIs here so the
            # Jinja2 template can render them as <img> tags or download links.
            # Place all attachment images inside  static/images/
            # -------------------------------------------------------------------
            "attachments": [
                resolve_attachment("sample1.png", static_dir),
                resolve_attachment("sample2.png", static_dir),
                resolve_attachment("sample3.png", static_dir),
            ],


            "approval": {
                "prepared_by":       "Jane Arianne A. Gaela",
                "reviewed_by_dev":   "Jomar Talambayan",
                "validated_by_qa":   "Joan Marie Peneda",
                "approved_by_owner": "Emmanuel Manaloto",
            },
        }
    ]

    html: str = template.render(
        css_path=css_file.resolve().as_uri(),
        logo_path=logo_file.resolve().as_uri(),

        generated_date="June 18, 2026",

        task={
            "client":        "Direc Business Technologies Inc.",
            "issue_logs_id": "EIL_TEST_001",
        },

        all_issue_titles=[
            "Attendance records not updating after biometric synchronization"
        ],

        assigned_consultant="Jane Arianne A. Gaela",

        approval_status="For Review",

        rca_sections=rca_sections,
    )

    html_file: Path = output_dir / "033.html"
    pdf_file:  Path = output_dir / "033.pdf"

    html_file.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch()

        page = browser.new_page()

        page.goto(
            html_file.resolve().as_uri(),
            wait_until="networkidle",
        )

        page.pdf(
            path=str(pdf_file),
            format="A4",
            print_background=True,
            # The CSS @page rule sets 1in on all sides.
            # Playwright's margin parameter is passed as 0mm here so that
            # the @page declaration is the single source of truth and
            # Playwright does not add a second margin on top of it.
            margin={
                "top":    "0mm",
                "bottom": "0mm",
                "left":   "0mm",
                "right":  "0mm",
            },
        )

        browser.close()

    print(f"HTML generated: {html_file}")
    print(f"PDF generated : {pdf_file}")


if __name__ == "__main__":
    main()