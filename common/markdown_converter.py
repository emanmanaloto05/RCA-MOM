"""
markdown_converter.py
----------------------
Converts the raw markdown text returned by the RCA generation agents
(Gemini) into safe, renderable HTML for the Jinja2 / WeasyPrint PDF
template (rca_template.html).

WHERE TO WIRE THIS IN
----------------------
Drop this file next to utils.py / graph.py and run it on each of the
six markdown-bearing RCA fields *before* they're placed into the
`rca_sections` list that gets passed to the template:

    cause, affected_module, impact_analysis,
    solution, preventive_action, owner_review

This should happen in build_pdf_section_dict() (or wherever that
function lives) -- ideally right after each agent's raw output is
pulled out of assembled_sections, and BEFORE any sentinel/error
string ("... generation unavailable", "... could not be generated")
is substituted in on a failure path. Converting the sentinel text
itself is harmless (verified below) but there's no reason to run it
through markdown for a string you wrote yourself.

USAGE
-----
    from markdown_converter import convert_rca_section_fields

    section = build_pdf_section_dict(assembled_sections, ...)
    section = convert_rca_section_fields(section)
    rca_sections.append(section)

or, for a single field:

    from markdown_converter import markdown_to_html
    section["cause"] = markdown_to_html(raw_cause_text)

WHY THE EXISTING SENTINEL CHECK STILL WORKS
--------------------------------------------
rca_template.html checks things like:
    {% if section.cause and "generation unavailable" not in section.cause.lower() %}

markdown.markdown() only wraps plain prose in <p> tags and applies
list/blockquote/etc. structure -- it does not rearrange or split
words that aren't using markdown syntax. So a sentinel string like
"Cause generation unavailable due to provider timeout." still
contains the contiguous substring "generation unavailable" after
conversion, and the check keeps working unchanged. (The only way
this would break is if the model wrapped those exact words in
markdown emphasis, e.g. "**generation** unavailable" -- not a
realistic case for a sentinel you control yourself.)
"""

from __future__ import annotations

from typing import Any, Optional

import markdown

_MD_EXTENSIONS: list[str] = [
    "extra",       # tables, fenced code blocks, footnotes, abbreviations, etc.
    "sane_lists",  # prevents adjacent ol/ul items from merging into one list
    "nl2br",       # turns single newlines into <br> -- LLM output is usually
                   # single-newline-separated rather than blank-line-separated,
                   # and without this, plain markdown collapses those into one
                   # run-on paragraph
]

# A single shared Markdown() instance is faster than re-instantiating per
# call, but it accumulates internal state (footnote refs, etc.) across
# conversions -- always call .reset() before each .convert() to make sure
# one section's content can't leak into the next.
#
# output_format: as of markdown>=3.0, "html5" is no longer a distinct
# option -- only "xhtml" (default) and "html" exist. "html" is the closer
# match (no self-closing void tags), and it's also the only value the
# installed type stub's Literal[...] accepts.
_MD = markdown.Markdown(extensions=_MD_EXTENSIONS, output_format="html")


def markdown_to_html(text: Optional[str]) -> str:
    """
    Convert a single markdown-formatted RCA section into HTML.

    Returns an empty string for None/empty/whitespace-only input so
    the template's truthiness checks (e.g. `{% if section.cause %}`)
    keep behaving the same way they do today.
    """
    if not text or not text.strip():
        return ""

    _MD.reset()
    return _MD.convert(text)


def convert_rca_section_fields(section: dict[str, Any]) -> dict[str, Any]:
    """
    Convenience helper: runs markdown_to_html() over every known
    markdown-bearing field in a single RCA section dict, in place,
    and returns the same dict so it can be used inline.

    Only touches fields that are actually present in the dict, so
    it's safe to call regardless of which optional fields a given
    section happens to have.
    """
    markdown_fields: tuple[str, ...] = (
        "cause",
        "affected_module",
        "impact_analysis",
        "solution",
        "preventive_action",
        "owner_review",
    )

    for field in markdown_fields:
        if field in section:
            section[field] = markdown_to_html(section.get(field))

    return section


if __name__ == "__main__":
    # Quick manual sanity check -- run `python markdown_converter.py`
    sample_section: dict[str, Optional[str]] = {
        "cause": (
            "**Root cause:** The approval workflow failed because the\n"
            "approval_status field was not refreshed after the trigger fired.\n\n"
            "- Missing refresh call in approve_record()\n"
            "- Race condition between cron job and manual approval\n"
        ),
        "solution": "Patch `approve_record()` to call `refresh_status()` before commit.",
        "affected_module": None,  # should become ""
    }
    converted: dict[str, Any] = convert_rca_section_fields(dict(sample_section))
    for key, value in converted.items():
        print(f"--- {key} ---")
        print(value)
        print()