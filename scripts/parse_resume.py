"""端到端解析一份真实的简历 PDF 并打印结构化结果。

用法（在项目根目录、已激活虚拟环境时）：

    python scripts/parse_resume.py path/to/resume.pdf

从 PDF 提取文本（确定性过程），并针对所配置的 LLM 提供商运行
解析智能体。以 JSON 形式打印结构化简历。绝不打印你的 API key。
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from multi_agent_resume_screener.agents.parser import parse_resume
from multi_agent_resume_screener.pdf import PDFExtractionError, extract_text_from_pdf
from multi_agent_resume_screener.state import ResumeRaw


def main(argv: list[str]) -> int:
    load_dotenv()

    if len(argv) < 2:
        print("Usage: python scripts/parse_resume.py <resume.pdf>")
        return 2

    pdf_path = Path(argv[1])
    try:
        text = extract_text_from_pdf(pdf_path)
    except (FileNotFoundError, PDFExtractionError) as exc:
        print(f"[PDF ERROR] {exc}")
        return 1

    print(f"Extracted {len(text)} characters from {pdf_path.name}\n")
    if not text.strip():
        print("No extractable text found (is this a scanned/image-only PDF?).")
        return 1

    print("Parsing with the configured LLM ...\n")
    structured = parse_resume(ResumeRaw(filename=pdf_path.name, text=text))
    print(structured.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
