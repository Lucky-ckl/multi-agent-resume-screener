"""PDF 提取与简历解析 agent 的测试。

这些是离线测试：解析器通过假 LLM 运行，因此不需要 API key 或网络。
真实的端到端解析通过 ``scripts/parse_resume.py`` 验证。
"""

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter

from multi_agent_resume_screener.agents.parser import parse_resume
from multi_agent_resume_screener.pdf import extract_text_from_pdf
from multi_agent_resume_screener.state import ResumeRaw, StructuredResume


# --------------------------------------------------------------------------- #
# 模仿 .with_structured_output(...).invoke(...) 接口的假 LLM
# --------------------------------------------------------------------------- #
class _FakeStructuredRunnable:
    def __init__(self, result, calls):
        self._result = result
        self._calls = calls

    def invoke(self, messages):
        self._calls.append(messages)
        return self._result


class _FakeLLM:
    """从 with_structured_output(...).invoke(...) 返回预设对象。"""

    def __init__(self, result):
        self.result = result
        self.calls: list = []

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredRunnable(self.result, self.calls)


# --------------------------------------------------------------------------- #
# PDF 提取
# --------------------------------------------------------------------------- #
def test_extract_text_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("definitely_not_a_real_file_12345.pdf")


def test_extract_text_rejects_bad_type():
    with pytest.raises(TypeError):
        extract_text_from_pdf(12345)  # type: ignore[arg-type]


def test_extract_text_from_blank_pdf_bytes():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)

    text = extract_text_from_pdf(buf.getvalue())
    assert isinstance(text, str)  # 空白页 -> 接近空的字符串，不崩溃


# --------------------------------------------------------------------------- #
# 解析器 agent
# --------------------------------------------------------------------------- #
def test_parse_resume_passes_text_and_returns_structured():
    expected = StructuredResume(name="Bob Builder", skills=["python", "fastapi"])
    llm = _FakeLLM(expected)

    out = parse_resume(
        ResumeRaw(filename="bob.pdf", text="Bob Builder\nSkills: python, fastapi"),
        llm=llm,
    )

    assert out.name == "Bob Builder"
    assert out.skills == ["python", "fastapi"]
    # 简历文本已被转发给模型（存在 human 消息）。
    assert llm.calls, "expected the LLM to be invoked"


def test_parse_resume_coerces_dict_result():
    llm = _FakeLLM({"name": "Dict Person", "skills": ["sql"]})
    out = parse_resume(ResumeRaw(filename="d.pdf", text="some text"), llm=llm)
    assert isinstance(out, StructuredResume)
    assert out.name == "Dict Person"
    assert out.skills == ["sql"]


def test_parse_resume_empty_text_skips_llm():
    class _BoomLLM:
        def with_structured_output(self, *args, **kwargs):
            raise AssertionError("LLM must not be called for empty input")

    out = parse_resume(ResumeRaw(filename="empty.pdf", text="   \n  "), llm=_BoomLLM())
    assert out == StructuredResume()
