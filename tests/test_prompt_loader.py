# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from agent_root.models import RCAInputModel
from common.utils import (
    PromptLoadError,
    PromptRenderError,
    RenderedPrompt,
    _load_yaml,
    _make_jinja_env,
    build_rca_chat_prompt,
    build_rca_template_context,
    get_prompt_templates,
    load_sys_prompt,
    reload_prompt_templates,
    render_rca_prompt,
)


def _write_yaml(path: Path, content: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(content), encoding="utf-8")


def _valid_prompts_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.yaml"
    _write_yaml(
        path,
        {
            "rca_generation": {
                "system": "You are a QA engineer.",
                "user": "Generate RCA for {{ issue_logs_id }}.",
            }
        },
    )
    return path


class TestLoadYaml:
    def test_returns_dict_for_valid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "test.yaml"
        _write_yaml(path, {"key": "value"})

        result = _load_yaml(path)

        assert result == {"key": "value"}

    def test_raises_if_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(PromptLoadError, match="not found"):
            _load_yaml(tmp_path / "missing.yaml")

    def test_raises_on_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("{ bad yaml: [unclosed", encoding="utf-8")

        with pytest.raises(PromptLoadError, match="Failed to parse"):
            _load_yaml(path)

    def test_raises_if_not_a_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(PromptLoadError, match="must be a YAML mapping"):
            _load_yaml(path)


class TestLoadSysPrompt:
    def test_loads_valid_file(self, tmp_path: Path) -> None:
        path = _valid_prompts_yaml(tmp_path)

        prompts = load_sys_prompt(path)

        assert "system" in prompts
        assert "user" in prompts
        assert prompts["system"] == "You are a QA engineer."

    def test_raises_missing_rca_generation_key(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.yaml"
        _write_yaml(path, {"other_key": {"system": "s", "user": "u"}})

        with pytest.raises(PromptLoadError, match="rca_generation"):
            load_sys_prompt(path)

    def test_raises_missing_system_key(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.yaml"
        _write_yaml(path, {"rca_generation": {"user": "Generate RCA."}})

        with pytest.raises(PromptLoadError, match="'system'"):
            load_sys_prompt(path)

    def test_raises_missing_user_key(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.yaml"
        _write_yaml(path, {"rca_generation": {"system": "System prompt."}})

        with pytest.raises(PromptLoadError, match="'user'"):
            load_sys_prompt(path)

    def test_raises_empty_system_prompt(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.yaml"
        _write_yaml(
            path,
            {"rca_generation": {"system": "   ", "user": "User prompt."}},
        )

        with pytest.raises(PromptLoadError, match="non-empty string"):
            load_sys_prompt(path)

    def test_raises_non_string_prompt(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.yaml"
        _write_yaml(path, {"rca_generation": {"system": 123, "user": "User."}})

        with pytest.raises(PromptLoadError, match="non-empty string"):
            load_sys_prompt(path)


class TestGetPromptTemplates:
    def test_returns_cached_dict(self) -> None:
        reload_prompt_templates()

        result1 = get_prompt_templates()
        result2 = get_prompt_templates()

        assert result1 is result2

    def test_cache_clears_on_reload(self) -> None:
        reload_prompt_templates()

        result1 = get_prompt_templates()
        reload_prompt_templates()
        result2 = get_prompt_templates()

        assert result1 == result2

    def test_contains_required_keys(self) -> None:
        prompts = get_prompt_templates()

        assert "system" in prompts
        assert "user" in prompts


class TestMakeJinjaEnv:
    def test_strict_undefined_raises_on_missing_var(self) -> None:
        from jinja2 import UndefinedError

        env = _make_jinja_env()
        template = env.from_string("Hello {{ missing_var }}")

        with pytest.raises(UndefinedError):
            template.render()

    def test_renders_known_variable(self) -> None:
        env = _make_jinja_env()
        template = env.from_string("Hello {{ name }}")

        assert template.render(name="World") == "Hello World"


class TestBuildRcaTemplateContext:
    def test_all_task_fields_present(self, rca_input: RCAInputModel) -> None:
        ctx = build_rca_template_context(rca_input)

        assert ctx["issue_logs_id"] == "EIL_TEST001"
        assert ctx["product"] == "Lotus"
        assert ctx["client"] == "TestClient"
        assert ctx["is_recurring"] is False

    def test_enum_values_are_strings(self, rca_input: RCAInputModel) -> None:
        ctx = build_rca_template_context(rca_input)

        assert isinstance(ctx["issue_type"], str)
        assert isinstance(ctx["urgency_level"], str)
        assert isinstance(ctx["priority_level"], str)

    def test_pr_fields_present_when_pr_supplied(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        ctx = build_rca_template_context(rca_input)

        assert ctx["pr_number"] == 42
        assert ctx["branch_name"] == "fix/approval-flow"
        assert isinstance(ctx["affected_modules"], list)

    def test_pr_fields_none_when_pr_absent(
        self,
        rca_input_minimal: RCAInputModel,
    ) -> None:
        ctx = build_rca_template_context(rca_input_minimal)

        assert ctx["pr_number"] is None
        assert ctx["branch_name"] is None
        assert ctx["affected_modules"] == []

    def test_dev_fields_present_when_dev_supplied(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        ctx = build_rca_template_context(rca_input)

        assert ctx["pic_dev"] == "Juan dela Cruz"
        assert ctx["dev_notes"] is not None

    def test_dev_fields_none_when_dev_absent(
        self,
        rca_input_minimal: RCAInputModel,
    ) -> None:
        ctx = build_rca_template_context(rca_input_minimal)

        assert ctx["pic_dev"] is None
        assert ctx["dev_notes"] is None

    def test_qa_fields_present_when_qa_supplied(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        ctx = build_rca_template_context(rca_input)

        assert ctx["pic_qa"] == "Maria Santos"
        assert ctx["fc_failed_testing"] == 0
        assert ctx["reopen_count"] == 0

    def test_qa_fields_default_when_qa_absent(
        self,
        rca_input_minimal: RCAInputModel,
    ) -> None:
        ctx = build_rca_template_context(rca_input_minimal)

        assert ctx["fc_failed_testing"] == 0
        assert ctx["existing_report"] is False
        assert ctx["pic_qa"] is None

    def test_dev_resolved_on_none_when_not_set(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        ctx = build_rca_template_context(rca_input)

        assert ctx["dev_resolved_on"] is None

    def test_qa_validated_on_none_when_not_set(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        ctx = build_rca_template_context(rca_input)

        assert ctx["qa_validated_on"] is None


class TestRenderedPrompt:
    def test_attributes_accessible(self) -> None:
        rendered_prompt = RenderedPrompt(system="sys", user="usr")

        assert rendered_prompt.system == "sys"
        assert rendered_prompt.user == "usr"

    def test_repr_shows_char_counts(self) -> None:
        rendered_prompt = RenderedPrompt(system="abc", user="de")

        assert "system_chars=3" in repr(rendered_prompt)
        assert "user_chars=2" in repr(rendered_prompt)


class TestRenderRcaPrompt:
    def test_returns_rendered_prompt(self, rca_input: RCAInputModel) -> None:
        rendered = render_rca_prompt(rca_input)

        assert isinstance(rendered, RenderedPrompt)
        assert len(rendered.system) > 0
        assert len(rendered.user) > 0

    def test_user_prompt_contains_issue_id(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        rendered = render_rca_prompt(rca_input)

        assert "EIL_TEST001" in rendered.user

    def test_raises_prompt_render_error_on_missing_variable(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        broken_templates = {
            "system": "System prompt.",
            "user": "Issue: {{ nonexistent_variable }}",
        }

        with patch("common.utils.get_prompt_templates", return_value=broken_templates):
            with pytest.raises(PromptRenderError, match="Template variable missing"):
                render_rca_prompt(rca_input)

    def test_raises_prompt_load_error_propagated(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        with patch(
            "common.utils.get_prompt_templates",
            side_effect=PromptLoadError("Simulated load failure"),
        ):
            with pytest.raises(PromptLoadError):
                render_rca_prompt(rca_input)


class TestBuildRcaChatPrompt:
    def test_returns_chat_prompt_template(self, rca_input: RCAInputModel) -> None:
        prompt = build_rca_chat_prompt(rca_input)

        assert isinstance(prompt, ChatPromptTemplate)

    def test_format_messages_returns_two_messages(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        prompt = build_rca_chat_prompt(rca_input)
        messages = prompt.format_messages()

        assert len(messages) == 2

    def test_first_message_is_system(self, rca_input: RCAInputModel) -> None:
        prompt = build_rca_chat_prompt(rca_input)
        messages = prompt.format_messages()

        assert isinstance(messages[0], SystemMessage)

    def test_second_message_is_human(self, rca_input: RCAInputModel) -> None:
        prompt = build_rca_chat_prompt(rca_input)
        messages = prompt.format_messages()

        assert isinstance(messages[1], HumanMessage)

    def test_human_message_contains_issue_id(
        self,
        rca_input: RCAInputModel,
    ) -> None:
        prompt = build_rca_chat_prompt(rca_input)
        messages = prompt.format_messages()

        content = getattr(messages[1], "content", "")
        content_text = content if isinstance(content, str) else str(content)

        assert "EIL_TEST001" in content_text