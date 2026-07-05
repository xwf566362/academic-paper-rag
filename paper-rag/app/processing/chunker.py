"""论文专属分片策略"""
import re
from typing import Dict, List, Optional, Tuple
from app.config import get_config
from app.processing.pdf_parser import parse_pdf, is_section_header


def build_chunks_from_pdf(file_path: str) -> List[Dict]:
    cfg = get_config()
    parsed = parse_pdf(file_path)
    full_text = parsed["full_text"]
    title = parsed["title"]
    file_name = parsed["file_name"]
    chunks: List[Dict] = []
    chunk_index = 0

    abstract_text = _extract_abstract(full_text)
    if abstract_text:
        chunks.append({"chunk_id": f"{file_name}_meta_{chunk_index}", "text": f"标题: {title}\n\n{abstract_text}",
                        "metadata": {"file_name": file_name, "paper_title": title, "section": "摘要", "chunk_index": chunk_index, "page_range": "1", "is_abstract": True}})
        chunk_index += 1

    sections = _split_by_sections(full_text)
    for sec_name, sec_text in sections:
        if _is_references_section(sec_name):
            continue
        sub_chunks = _split_section_into_chunks(text=sec_text, section=sec_name, max_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        for sub in sub_chunks:
            chunks.append({"chunk_id": f"{file_name}_sec_{chunk_index}", "text": sub["text"],
                            "metadata": {"file_name": file_name, "paper_title": title, "section": sec_name, "chunk_index": chunk_index, "page_range": sub.get("page_range", ""), "is_abstract": False}})
            chunk_index += 1

    refs = _extract_reference_text(full_text)
    if refs:
        for ref_text in refs:
            chunks.append({"chunk_id": f"{file_name}_ref_{chunk_index}", "text": ref_text,
                            "metadata": {"file_name": file_name, "paper_title": title, "section": "参考文献", "chunk_index": chunk_index, "page_range": "", "is_reference": True}})
            chunk_index += 1
    return chunks


def _extract_abstract(text: str) -> Optional[str]:
    lines = text.split("\n")
    abstract_start = -1
    for i, line in enumerate(lines):
        if re.match(r"^(摘要|abstract)\s*$", line.strip(), re.IGNORECASE):
            abstract_start = i
            break
    if abstract_start < 0:
        return None
    keywords_found = re.compile(r"^(关键词|keywords)\s*[:：]?\s*", re.IGNORECASE)
    section_found = re.compile(r"^(?:第[一二三四五六七八九十]+章\s*)?(?:引言|绪论|introduction|background|1\.?\s*引言?)", re.IGNORECASE)
    abstract_end = -1
    for i in range(abstract_start + 1, min(abstract_start + 200, len(lines))):
        stripped = lines[i].strip()
        if keywords_found.match(stripped):
            abstract_end = i + 1
            break
        if section_found.match(stripped):
            abstract_end = i
            break
    if abstract_end < 0:
        abstract_end = min(abstract_start + 100, len(lines))
    result = "\n".join(lines[abstract_start:abstract_end]).strip()
    return result if len(result) > 20 else None


def _split_by_sections(text: str) -> List[Tuple[str, str]]:
    lines = text.split("\n")
    sections: List[Tuple[str, str]] = []
    current_sec = "前言"
    current_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)
            continue
        if is_section_header(stripped):
            if current_lines:
                sections.append((current_sec, "\n".join(current_lines).strip()))
            current_sec = stripped
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_sec, "\n".join(current_lines).strip()))
    return sections


def _is_references_section(name: str) -> bool:
    return bool(re.match(r"^(参考文献|references|bibliography)\s*$", name.strip(), re.IGNORECASE))


def _split_section_into_chunks(text: str, section: str, max_size: int, overlap: int) -> List[Dict]:
    if not text.strip():
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[Dict] = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 1 <= max_size:
            current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para
        else:
            if current_chunk.strip():
                chunks.append({"text": f"[{section}]\n{current_chunk.strip()}", "page_range": ""})
            if len(para) > max_size:
                sub_start = 0
                while sub_start < len(para):
                    sub_end = min(sub_start + max_size, len(para))
                    chunks.append({"text": f"[{section}]\n{para[sub_start:sub_end].strip()}", "page_range": ""})
                    sub_start = sub_end - overlap if sub_end < len(para) else sub_end
                current_chunk = ""
            else:
                current_chunk = para
    if current_chunk.strip():
        chunks.append({"text": f"[{section}]\n{current_chunk.strip()}", "page_range": ""})
    return chunks


def _extract_reference_text(text: str) -> List[str]:
    lines = text.split("\n")
    ref_start = -1
    for i, line in enumerate(lines):
        if re.match(r"^(?:参考文献|references|bibliography)\s*$", line.strip(), re.IGNORECASE):
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