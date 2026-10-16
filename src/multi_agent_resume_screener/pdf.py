"""确定性的 PDF -> 文本提取。

这是一个普通函数，不是智能体：把 PDF 转成文本不需要任何判断，
因此我们用 ``pypdf`` 让这一步保持廉价且可复现。产出的文本随后交给
parser *智能体* 转换成结构化字段。
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


class PDFExtractionError(RuntimeError):
    """当 PDF 无法读取或无法提取文本时抛出。"""


def extract_text_from_pdf(source: str | Path | bytes) -> str:
    """从 PDF 中提取纯文本，接受文件路径或原始字节。

    Args:
        source: PDF 文件的路径（``str``/``Path``），或 PDF 的原始字节
            （例如文件上传接收到的内容）。

    Returns:
        所有页面拼接而成的文本，多余的空行会被压缩。
        对于扫描件/纯图片 PDF，可能返回空字符串。

    Raises:
        FileNotFoundError: 给定了路径但该位置没有文件。
        TypeError: ``source`` 既不是路径也不是字节。
        PDFExtractionError: PDF 格式损坏或无法读取。
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        reader_source: str | BytesIO = str(path)
    elif isinstance(source, (bytes, bytearray)):
        reader_source = BytesIO(bytes(source))
    else:
        raise TypeError(
            f"Expected a path or bytes, got {type(source).__name__}."
        )

    try:
        reader = PdfReader(reader_source)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - 清晰地暴露 pypdf 的任何失败
        raise PDFExtractionError(f"Failed to read PDF: {exc}") from exc

    text = "\n".join(pages)
    # 把连续 3 个以上的换行压缩为一个空行分隔。
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
