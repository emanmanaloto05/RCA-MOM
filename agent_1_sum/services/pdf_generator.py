from pathlib import Path

import pdfkit


WKHTMLTOPDF_PATH = (
    r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
)


def generate_pdf_from_html(
    html_path: str,
    file_name: str
) -> str:
    output_dir = Path(
        "agent_1_sum/output/pdf"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_path = output_dir / file_name

    config = pdfkit.configuration(
        wkhtmltopdf=WKHTMLTOPDF_PATH
    )

    options = {
        "page-size": "A4",
        "encoding": "UTF-8",
        "enable-local-file-access": None,
        "print-media-type": None,
        "margin-top": "20mm",
        "margin-right": "20mm",
        "margin-bottom": "20mm",
        "margin-left": "20mm",
    }

    pdfkit.from_file(
        html_path,
        str(pdf_path),
        configuration=config,
        options=options
    )

    return str(
        pdf_path
    )