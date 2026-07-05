import sys, os
sys.path.insert(0, ".")

lines = open("app/api/paper_analyzer.py", "r", encoding="utf-8").read().split("\n")

# 找到 get_cached_paper 函数，添加磁盘回退
for i, line in enumerate(lines):
    if "def get_cached_paper" in line:
        old = "\n".join(lines[i:i+4])
        new = (
            'def get_cached_paper(file_name: str) -> Optional[str]:\n'
            '    """获取缓存的论文全文，如内存中无则从磁盘读取"""\n'
            '    text = _paper_cache.get(file_name)\n'
            '    if text:\n'
            '        return text\n'
            '    # 缓存丢失时尝试从磁盘读取\n'
            '    from app.config import PAPERS_DIR\n'
            '    pdf_path = PAPERS_DIR / file_name\n'
            '    if pdf_path.exists():\n'
            '        try:\n'
            '            from app.processing.pdf_parser import parse_pdf\n'
            '            parsed = parse_pdf(str(pdf_path))\n'
            '            _paper_cache[file_name] = parsed["full_text"]\n'
            '            return parsed["full_text"]\n'
            '        except Exception as e:\n'
            '            logger.warning(f"从磁盘恢复缓存失败: {e}")\n'
            '    return None\n'
        )
        lines = lines[:i] + new.split("\n") + lines[i+4:]
        break

open("app/api/paper_analyzer.py", "w", encoding="utf-8").write("\n".join(lines))
compile("\n".join(lines), "", "exec")
print("Cache persistence OK")