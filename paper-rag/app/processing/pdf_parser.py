"""PDF解析模块"""
import re
import fitz
from pathlib import Path
from typing import Dict, List, Optional

SECTION_PATTERNS = [
    r"^(?:第[一二三四五六七八九十]+章\s*)?(?:摘要|abstract)\s*$",
    r"^(?:第[一二三四五六七八九十]+章\s*)?(?:引言|绪论|前言|introduction|background)\s*$",
    r"^(?:第[一二三四五六七八九十]+章\s*)?(?:相关工作|related\s*work|文献综述|literature\s*review)\s*$",
    r"^(?:第[一二三四五六七八九十]+章\s*)?(?:方法|methodology|方法论|approach|method|模型|model|算法|algorithm)\s*$",
    r"^(?:第[一二三四五六七八九十]+章\s*)?(?:实验|experiment|评估|evaluation|结果|results|分析|analysis|讨论|discussion)\s*$",
    r"^(?:第[一二三四五六七八九十]+章\s*)?(?:结论|conclusion|总结|summary|future\s*work|展望)\s*$",
    r"^(?:参考文献|references|bibliography)\s*$",
    r"^(?:附录|appendix)\s*[A-Z]?\s*$",
]

_section_regexes = [re.compile(p, re.IGNORECASE) for p in SECTION_PATTERNS]


def is_section_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 100 or line.endswith((".", "。", ":", "：", ";", "；")):
        return False
    for r in _section_regexes:
        if r.match(line):
            return True
    if re.match(r"^(\d+\.)+\d*\s+", line):
        return True
    return False


def parse_pdf(file_path: str) -> Dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {file_path}")
    doc = fitz.open(file_path)
    meta = doc.metadata
    page_count = doc.page_count
    sections = []
    full_text_parts = []
    current_section = "前言"
    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text("text")
        if not text.strip():
            continue
        lines = text.split("\n")
        page_blocks = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if is_section_header(stripped):
                current_section = stripped
            page_blocks.append({"text": stripped, "section": current_section})
        sections.append({"page_num": page_num + 1, "blocks": page_blocks})
        full_text_parts.append(text)
    doc.close()
    full_text = "\n".join(full_text_parts)
    title = path.stem
    if sections and sections[0]["blocks"]:
        candidate = sections[0]["blocks"][0]["text"]
        if 5 < len(candidate) < 200 and not candidate.endswith((".", "。", ":", "：", "!")):
            title = candidate
    return {"file_name": path.name, "file_path": str(path), "title": title, "page_count": page_count,
            "metadata": {"author": meta.get("author", ""), "subject": meta.get("subject", ""), "keywords": meta.get("keywords", "")},
            "sections": sections, "full_text": full_text}


def extract_references(text: str) -> List[str]:
    lines = text.split("\n")
    ref_start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^(?:参考文献|references|bibliography)\s*$", stripped, re.IGNORECASE):
            ref_start = i
            break
    if ref_start < 0:
        return []
    refs = []
    for line in lines[ref_start + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\[\d+\]|^\d+\.", stripped):
            refs.append(stripped)
        elif refs:
            refs[-1] += " " + stripped
    return refs