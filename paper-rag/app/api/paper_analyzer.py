"""
论文分析引擎
提供：方法结构化解析、算法流程图生成、论文全文缓存
专项适配 CT-Font 汉字书法风格迁移领域
"""

import logging
from typing import Dict, List, Optional

from app.api.llm_client import ACADEMIC_SYSTEM_PROMPT, get_llm_client

logger = logging.getLogger(__name__)


# ============================================================
# 论文全文缓存（避免重复读取文件）
# ============================================================

_paper_cache: Dict[str, str] = {}


def cache_paper(file_name: str, full_text: str):
    """缓存已解析的论文全文"""
    _paper_cache[file_name] = full_text
    logger.info(f"缓存论文: {file_name} ({len(full_text)} 字符)")


def get_cached_paper(file_name: str) -> Optional[str]:
    """获取缓存的论文全文，如内存中无则从磁盘读取"""
    text = _paper_cache.get(file_name)
    if text:
        return text
    # 缓存丢失时尝试从磁盘读取
    from app.config import PAPERS_DIR
    pdf_path = PAPERS_DIR / file_name
    if pdf_path.exists():
        try:
            from app.processing.pdf_parser import parse_pdf
            parsed = parse_pdf(str(pdf_path))
            _paper_cache[file_name] = parsed["full_text"]
            return parsed["full_text"]
        except Exception as e:
            logger.warning(f"从磁盘恢复缓存失败: {e}")
    return None


def clear_paper_cache(file_name: str):
    """清除指定论文缓存"""
    _paper_cache.pop(file_name, None)


# ============================================================
# AI 提示词模板
# ============================================================

METHOD_ANALYSIS_PROMPT = """你是一位资深 AI 研究助理，专门从事计算机视觉与字体生成方向论文的方法论解析。

## 任务
根据提供的论文全文，按以下固定模块顺序输出结构化研究方法解析。

## 输出格式（必须严格按顺序）

### 1. 研究问题与核心创新点
- 论文要解决的核心问题
- 提出的主要创新点（2-4点）
- 与现有工作的核心区别

### 2. 数据集构建、评价指标与实验环境
- 使用数据集（名称、规模、来源）
- 数据预处理方式
- 评价指标
- 实验软硬件环境（GPU、框架等）

### 3. 完整技术链路
数据预处理 → 主干网络/核心算法 → 各注意力模块/融合模块 → 损失函数设计 → 优化器训练策略

每个环节需描述其输入、处理逻辑、输出。

### 4. 对比基线模型与消融实验
- 对比的基线模型
- 消融实验设置
- 各模块贡献分析

### 5. 本方法局限性与不足
- 论文承认的局限性
- 实验未覆盖的场景
- 可能的改进方向

## 专项规则（CT-Font 汉字书法风格迁移领域）
如果论文涉及汉字书法/字体生成/风格迁移，必须优先识别并重点提取：
1. 汉字书法数据集（字体种类、字符数、风格标签）
2. SKA 选择性核注意力模块
3. SCCA 风格通道注意力模块
4. DSA 可变形空间注意力模块
5. 多尺度特征融合策略
6. 三级协同注意力机制
7. 书法生成专属损失函数
8. 机器人书法实验设置

## 约束
- 严格基于论文原文内容，不编造
- 如果论文无完整算法/方法章节，在输出开头注明
- 格式清晰，适合 SCI 论文写作阅读习惯
- 每个模块使用完整段落描述，避免碎片化列表"""


FLOWCHART_PROMPT = """你是一位 AI 算法分析专家，专门从论文方法描述中提取网络结构并生成规范的 Mermaid 流程图。

## 任务
根据提供的论文全文，提取和分析论文中的算法流程与网络结构：
1. 识别并提取论文核心算法/网络结构
2. 拆分算法层级模块（输入、特征提取、各分支、融合、输出、损失）
3. 梳理并行/串行/残差/多分支数据流转关系
4. 生成 Mermaid 流程图代码作为可视化辅助

## Mermaid 流程图规范
- 使用 graph TB 方向（从上到下）
- 用方框 [] 表示处理模块
- 用圆角框 () 表示输入/输出
- 用菱形 {} 表示条件/判断
- 子图 subgraph 表示模块分组
- 用 --> 连接数据流向
- 用 -.-> 连接辅助信息流
- 并行分支用多个箭头从同一节点出发

## 输出格式
### 第一部分：算法流程分析
按以下结构分析论文中的算法：
1. 算法/网络整体架构：概述核心设计思路
2. 输入与数据准备：输入格式、预处理方式
3. 特征提取：主干网络/编码器设计
4. 核心模块拆解：各分支的功能与连接关系
5. 训练与优化：损失函数、优化器、训练策略
6. 各模块串/并行关系总结

### 第二部分：Mermaid 流程图代码（可选可视化）

## 专项适配 CT-Font 模型
如果识别到 CT-Font/汉字书法生成模型，自动：
- 识别三级协同注意力架构
- 拆分 SKA/SCCA/DSA 多分支
- 标注可变形特征融合链路
- 区分风格编码与内容编码

文字分析在前，Mermaid 代码在后，用 ===MERMAID=== 分隔。

## Mermaid 语法注意事项（必须遵守）
1. 图方向用 graph TB（从上到下）
2. 节点ID必须用英文字母开头，不能以数字开头
3. 节点标签放在方括号或圆括号中，如 A[输入图像] 或 B(输出结果)
4. 连接用 -->
5. 子图用 subgraph...end
6. 严禁使用圆括号作为标签的一部分（如 Input(图像) 应改为 Input[图像]）
7. 每个节点先定义再引用，避免悬空引用
8. 最终输出的 Mermaid 代码必须符合标准语法，可直接渲染"""


# ============================================================
# 分析方法解析
# ============================================================


def _split_text_into_chunks(text: str, chunk_size: int = 4000) -> list:
    """将长文本分割为多个片段，每个约 chunk_size 字符"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):

            # 尽量在段落边界处分割
            cut = text.rfind("\n\n", start, end)
            if cut > start + chunk_size // 2:
                end = cut + 2
            else:
                cut = text.rfind("\n", start, end)
                if cut > start + chunk_size // 2:
                    end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

async def analyze_method(file_name: str, full_text: str, provider: Optional[str] = None) -> Dict:
    """
    对论文全文进行结构化方法解析
    Returns:
        {
            "file_name": str,
            "analysis": str (结构化分析Markdown文本),
            "has_method_section": bool,
            "is_ct_font": bool (是否汉字书法相关)
        }
    """
    # 检测是否 CT-Font 相关
    is_ct_font = any(kw in full_text[:5000].lower() for kw in [
        "ct-font", "书法", "calligraphy", "font generation", "style transfer",
        "汉字", "chinese character", "glyph", "stroke", "骨架", "skeleton"
    ])

    prompt = METHOD_ANALYSIS_PROMPT
    if is_ct_font:
        prompt += (
            "\n\n## 额外说明\n"
            "检测到本文属于汉字书法/字体生成领域，请严格按照 CT-Font 专项规则优先提取相关内容，"
            "重点分析三级协同注意力、多分支注意力融合、书法风格编码等专有模块。"
        )

    has_method = any(kw in full_text.lower() for kw in [
        "method", "approach", "architecture", "network", "framework",
        "方法", "架构", "网络结构", "算法"
    ])

    # 将全文分成多个大块传入
    text_chunks = _split_text_into_chunks(full_text, 4000)
    context_chunks = [
        {"text": c, "metadata": {"file_name": file_name, "part": i+1}}
        for i, c in enumerate(text_chunks)
    ]

    try:
        client = get_llm_client(provider)
        result = await client.chat(
            system_prompt=ACADEMIC_SYSTEM_PROMPT,
            context_chunks=context_chunks,
            user_query=prompt,
            max_chunk_size=4000,
            max_tokens=4096,
        )
    except Exception as e:
        logger.error(f"方法解析失败: {e}")
        raise

    return {
        "file_name": file_name,
        "analysis": result,
        "has_method_section": has_method,
        "is_ct_font": is_ct_font,
    }


async def generate_flowchart(file_name: str, full_text: str, provider: Optional[str] = None) -> Dict:
    """从论文方法描述生成算法流程图"""
    has_method = any(kw in full_text.lower() for kw in [
        "method", "approach", "architecture", "network", "framework",
        "方法", "架构", "网络结构", "算法"
    ])

    # 将全文分成多个大块
    text_chunks = _split_text_into_chunks(full_text, 4000)
    context_chunks = [
        {"text": c, "metadata": {"file_name": file_name, "part": i+1}}
        for i, c in enumerate(text_chunks)
    ]

    try:
        client = get_llm_client(provider)
        result = await client.chat(
            system_prompt=ACADEMIC_SYSTEM_PROMPT,
            context_chunks=context_chunks,
            user_query=FLOWCHART_PROMPT,
            max_chunk_size=4000,
            max_tokens=4096,
        )
    except Exception as e:
        logger.error(f"流程图生成失败: {e}")
        raise

    steps, mermaid_code = _parse_flowchart_result(result)

    return {
        "file_name": file_name,
        "steps": steps,
        "mermaid_code": mermaid_code,
        "has_method_section": has_method,
    }


def _parse_flowchart_result(result: str) -> tuple:
    """解析AI返回结果，分离文字步骤和Mermaid代码"""
    steps, mermaid_code = "", ""
    if "===MERMAID===" in result:
        parts = result.split("===MERMAID===", 1)
        steps = parts[0].strip()
        mermaid_code = parts[1].strip()
    else:
        import re
        mermaid_match = re.search(r"```mermaid\s*\n(.*?)\n```", result, re.DOTALL)
        if mermaid_match:
            idx = result.find("```mermaid")
            steps = result[:idx].strip()
            mermaid_code = mermaid_match.group(1).strip()
        else:
            code_match = re.search(r"```\s*\n(.*?)\n```", result, re.DOTALL)
            if code_match:
                idx = result.find("```")
                steps = result[:idx].strip()
                mermaid_code = code_match.group(1).strip()
            else:
                steps = result

    # 清理和验证 Mermaid 代码
    if mermaid_code:
        mermaid_code = _clean_mermaid_code(mermaid_code)

    return steps, mermaid_code


def _clean_mermaid_code(code: str) -> str:
    """清理 Mermaid 代码，修复常见语法问题"""
    if not code:
        return ""
    import re
    # 确保有 graph 声明
    if not code.strip().startswith("graph"):
        code = "graph TB\n" + code
    # 移除代码块标记（如果还有残留）
    code = re.sub(r"^```(?:mermaid)?\s*", "", code, flags=re.MULTILINE)
    code = code.replace("```", "")
    # 修复节点ID（确保以字母开头）
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("%%"):
            cleaned.append(line)
            continue
        # 修复行首数字开头的节点ID (如 "1[dog]" -> "N1[dog]")
        cleaned.append(line)
    return "\n".join(cleaned).strip()
