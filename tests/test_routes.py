# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from httpx import Response

from agent_root.models import RCAInputModel, RCAOutputModel
from agent_root.service import RCAServiceError
from app import app
from config.settings import settings


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {
        settings.api_key_header: settings.api_key,
    }


def _payload(rca_input: RCAInputModel) -> dict[str, Any]:
    return rca_input.model_dump(mode="json")


def test_generate_rca_requires_api_key(
    rca_input: RCAInputModel,
) -> None:
    response = cast(
        Response,
        client.post(
            "/api/generate-rca",
            json=_payload(rca_input),
        ),
    )

    assert response.status_code == 401


def test_generate_rca_rejects_invalid_api_key(
    rca_input: RCAInputModel,
) -> None:
    response = cast(
        Response,
        client.post(
            "/api/generate-rca",
            headers={settings.api_key_header: "wrong-api-key"},
            json=_payload(rca_input),
        ),
    )

    assert response.status_code == 403


def test_generate_rca_rejects_invalid_payload() -> None:
    response = cast(
        Response,
        client.post(
            "/api/generate-rca",
            headers=_auth_headers(),
            json={},
        ),
    )

    assert response.status_code == 422


def test_generate_rca_success(
    rca_input: RCAInputModel,
    valid_markdown: str,
) -> None:
    expected_output = RCAOutputModel(
        issue_id="EIL_TEST001",
        markdown_rca=valid_markdown,
        pdf_file_path=None,
    )

    with patch(
        "api.routes.RCAService.generate_rca",
        new=AsyncMock(return_value=expected_output),
    ):
        response = cast(
            Response,
            client.post(
                "/api/generate-rca",
                headers=_auth_headers(),
                json=_payload(rca_input),
            ),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["issue_id"] == "EIL_TEST001"
    assert "## 1. Issue Summary" in body["markdown_rca"]


def test_generate_rca_service_error_returns_500(
    rca_input: RCAInputModel,
) -> None:
    with patch(
        "api.routes.RCAService.generate_rca",
        new=AsyncMock(side_effect=RCAServiceError("Generation failed")),
    ):
        response = cast(
            Response,
            client.post(
                "/api/generate-rca",
                headers=_auth_headers(),
                json=_payload(rca_input),
            ),
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to generate RCA. Please check server logs."