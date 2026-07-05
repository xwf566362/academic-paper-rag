"""Gradio 前端界面"""
import logging
from pathlib import Path
from typing import Tuple
import gradio as gr
import httpx
from app.config import get_config

logger = logging.getLogger(__name__)
API_BASE = "http://127.0.0.1:8000/api"


async def api_post(path: str, data: dict) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{API_BASE}{path}", json=data)
    resp.raise_for_status()
    return resp.json()


async def api_get(path: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{API_BASE}{path}")
    resp.raise_for_status()
    return resp.json()


async def api_upload(path: str, file_path: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        with open(file_path, "rb") as f:
            resp = await client.post(f"{API_BASE}{path}", files={"file": (Path(file_path).name, f, "application/pdf")})
    resp.raise_for_status()
    return resp.json()


async def api_delete(path: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.delete(f"{API_BASE}{path}")
    resp.raise_for_status()
    return resp.json()


async def ask_question(question: str, paper_filter: str, history: list) -> Tuple[list, str]:
    if not question.strip():
        return history, ""
    req = {"question": question}
    if paper_filter and paper_filter != "\u5168\u90e8":
        req["paper_filter"] = paper_filter
    try:
        result = await api_post("/ask", req)
        answer = result.get("answer", "\u65e0\u56de\u7b54")
        sources = result.get("sources", [])
        source_text = ""
        if sources:
            source_text = "\n\n---\n**\u8bba\u6587\u6eaf\u6e90**\n"
            for i, s in enumerate(sources, 1):
                source_text += f"\n{i}. \u8bba\u6587: {s.get('paper', '\u672a\u77e5')} | \u7ae0\u8282: {s.get('section', '\u672a\u77e5')} | \u76f8\u5173\u5ea6: {s.get('score', 0):.3f}"
        full_answer = answer + source_text
    except httpx.ConnectError:
        full_answer = "\u65e0\u6cd5\u8fde\u63a5\u5230\u540e\u7aef\u670d\u52a1\u3002"
    except Exception as e:
        full_answer = f"\u8bf7\u6c42\u5931\u8d25: {e}"
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_answer})
    return history, ""


async def compare_papers(question: str, paper_1: str, paper_2: str, history: list) -> Tuple[list, str]:
    if not question.strip():
        return history, ""
    if not paper_1 or not paper_2:
        history.append({ "role": "user", "content": question })
        history.append({ "role": "assistant", "content": "请分别选择两篇论文。"})
        return history, ""
    req = { "question": question, "paper_names": [paper_1, paper_2] }
    try:
        result = await api_post("/ask/compare", req)
        answer = result.get("answer", "\u65e0\u56de\u7b54")
        sources = result.get("sources", [])
        source_text = ""
        seen = set()
        if sources:
            source_text = "\n\n---\n**\u8bba\u6587\u6eaf\u6e90**\n"
            for i, s in enumerate(sources, 1):
                key = f"{s.get('paper', '')}/{s.get('section', '')}"
                if key not in seen:
                    seen.add(key)
                    source_text += f"\n{i}. \u8bba\u6587: {s.get('paper', '\u672a\u77e5')} | \u7ae0\u8282: {s.get('section', '\u672a\u77e5')} | \u76f8\u5173\u5ea6: {s.get('score', 0):.3f}"
        full_answer = answer + source_text
    except httpx.ConnectError:
        full_answer = "\u65e0\u6cd5\u8fde\u63a5\u5230\u540e\u7aef\u670d\u52a1\u3002"
    except Exception as e:
        full_answer = f"\u8bf7\u6c42\u5931\u8d25: {e}"
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_answer})
    return history, ""


async def refresh_papers() -> str:
    try:
        papers = await api_get("/papers")
        if not papers:
            return "\u77e5\u8bc6\u5e93\u4e3a\u7a7a\u3002"
        lines = ["| \u8bba\u6587\u540d | \u6807\u9898 | \u5206\u7247\u6570 |", "|---|---|---|"]
        for p in papers:
            lines.append(f"| {p['file_name']} | {p['paper_title'][:40]} | {p['chunk_count']} |")
        return "\n".join(lines)
    except Exception as e:
        return f"\u83b7\u53d6\u5931\u8d25: {e}"


async def refresh_stats() -> str:
    try:
        stats = await api_get("/stats")
        return f"\u8bba\u6587\u603b\u6570: {stats['paper_count']} | \u5206\u7247: {stats['total_chunks']} | \u9ed8\u8ba4LLM: {stats['default_provider']} | \u53ef\u7528: {', '.join(stats['available_providers'])}"
    except Exception as e:
        return f"\u83b7\u53d6\u5931\u8d25: {e}"


async def upload_pdf(file):
    if file is None:
        return "\u8bf7\u9009\u62e9PDF\u6587\u4ef6"
    file_path = str(file)
    try:
        result = await api_upload("/upload", file_path)
        return f"\u4e0a\u4f20\u6210\u529f: {result['file_name']} ({result['chunks']} \u4e2a\u5206\u7247)"
    except Exception as e:
        return f"\u4e0a\u4f20\u5931\u8d25: {e}"


async def delete_paper(file_name: str) -> str:
    if not file_name.strip():
        return "\u8bf7\u8f93\u5165\u8bba\u6587\u6587\u4ef6\u540d"
    try:
        result = await api_delete(f"/papers/{file_name.strip()}")
        return f"\u5220\u9664\u6210\u529f: {result['message']}"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"\u672a\u627e\u5230\u8bba\u6587: {file_name}"
        return f"\u5220\u9664\u5931\u8d25: {e}"
    except Exception as e:
        return f"\u5220\u9664\u5931\u8d25: {e}"



from typing import Tuple

async def analyze_paper_method(paper_name: str, provider: str):
    if not paper_name:
        return "请先上传并解析论文。"
    import httpx
    try:
        req = {"file_name": paper_name}
        if provider and provider != "默认":
            req["provider"] = provider
        r = await httpx.AsyncClient(timeout=180).post("http://127.0.0.1:8000/api/analyze/method", json=req)
        r.raise_for_status()
        data = r.json()
        result = data.get("analysis", "")
        if not data.get("has_method_section"):
            result = "> 本文无完整算法与方法章节，仅提取现有可识别内容。\n\n" + result
        return result
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            return "请先完成论文上传与全文解析。"
        return f"分析失败: {e}"
    except Exception as e:
        return f"请求失败: {e}"


async def generate_paper_flowchart(paper_name: str, provider: str):
    if not paper_name:
        return "请先上传并解析论文。"
    import httpx
    try:
        req = {"file_name": paper_name}
        if provider and provider != "默认":
            req["provider"] = provider
        r = await httpx.AsyncClient(timeout=180).post("http://127.0.0.1:8000/api/analyze/flowchart", json=req)
        r.raise_for_status()
        data = r.json()
        steps = data.get("steps", "")
        code = data.get("mermaid_code", "")
        if not data.get("has_method_section"):
            steps = "> 本文无完整算法与方法章节，仅提取现有可识别内容。\n\n" + steps
        result = steps
        if code:
            result += "\n\n---\n## Mermaid 流程图代码\n```mermaid\n" + code + "\n```"
        return result
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            return "请先完成论文上传与全文解析。"
        return f"分析失败: {e}"
    except Exception as e:
        return f"请求失败: {e}"

def _get_paper_choices():
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:8000/api/papers", timeout=10)
        return [p["file_name"] for p in r.json()]
    except:
        return []


def build_app() -> gr.Blocks:
    cfg = get_config()
    providers = ["\u9ed8\u8ba4"] + list(cfg.llm_providers.keys())
    with gr.Blocks(title="\u8bba\u6587RAG\u77e5\u8bc6\u5e93") as app:
        gr.Markdown("# \u8bba\u6587RAG\u77e5\u8bc6\u5e93\n\u57fa\u4e8e\u672c\u5730\u5d4c\u5165 + \u8fdc\u7a0bLLM\u7684\u5b66\u672f\u8bba\u6587\u667a\u80fd\u95ee\u7b54\u7cfb\u7edf\u3002")
        with gr.Tabs():
            with gr.TabItem("\u8bba\u6587\u95ee\u7b54"):
                chatbot = gr.Chatbot(label="\u5bf9\u8bdd\u8bb0\u5f55", height=500)
                with gr.Row():
                    msg = gr.Textbox(label="\u8f93\u5165\u95ee\u9898", placeholder="\u8bf7\u8f93\u5165\u5173\u4e8e\u8bba\u6587\u7684\u95ee\u9898...", scale=4)
                    submit_btn = gr.Button("\u53d1\u9001", variant="primary", scale=1)
                with gr.Row():
                    paper_filter = gr.Dropdown(label="\u9650\u5b9a\u8bba\u6587", choices=["\u5168\u90e8"], value="\u5168\u90e8", interactive=True, scale=2)
                    clear_btn = gr.Button("\u6e05\u7a7a\u5bf9\u8bdd", size="sm", scale=1)
                submit_btn.click(ask_question, inputs=[msg, paper_filter, chatbot], outputs=[chatbot, msg])
                msg.submit(ask_question, inputs=[msg, paper_filter, chatbot], outputs=[chatbot, msg])
                clear_btn.click(lambda: [], outputs=[chatbot])

            with gr.TabItem("\u591a\u8bba\u6587\u5bf9\u6bd4"):
                compare_chatbot = gr.Chatbot(label="\u5bf9\u6bd4\u5bf9\u8bdd", height=450)
                with gr.Row():
                    compare_question = gr.Textbox(label="\u5bf9\u6bd4\u95ee\u9898", placeholder="\u8f93\u5165\u8981\u5bf9\u6bd4\u5206\u6790\u7684\u95ee\u9898...", scale=3)
                    compare_paper_1 = gr.Dropdown(label="第一篇论文", choices=_get_paper_choices(), interactive=True, scale=1)
                    compare_paper_2 = gr.Dropdown(label="第二篇论文", choices=_get_paper_choices(), interactive=True, scale=1)
                with gr.Row():
                    compare_btn = gr.Button("\u5bf9\u6bd4\u5206\u6790", variant="primary", scale=1)
                    clear_compare_btn = gr.Button("\u6e05\u7a7a", size="sm", scale=1)
                compare_btn.click(compare_papers, inputs=[compare_question, compare_paper_1, compare_paper_2, compare_chatbot], outputs=[compare_chatbot, compare_question])
                clear_compare_btn.click(lambda: [], outputs=[compare_chatbot])

            with gr.TabItem("\u8bba\u6587\u7ba1\u7406"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### \u4e0a\u4f20\u8bba\u6587")
                        pdf_upload = gr.File(label="\u9009\u62e9PDF\u6587\u4ef6", file_types=[".pdf"], file_count="single")
                        upload_status = gr.Markdown("")
                    with gr.Column():
                        gr.Markdown("### \u5220\u9664\u8bba\u6587")
                        delete_input = gr.Textbox(label="\u8bba\u6587\u6587\u4ef6\u540d", placeholder="example.pdf")
                        delete_btn = gr.Button("\u5220\u9664", variant="stop")
                        delete_status = gr.Markdown("")
                gr.Markdown("---")
                with gr.Row():
                    refresh_list_btn = gr.Button("\u5237\u65b0\u8bba\u6587\u5217\u8868", variant="secondary")
                    refresh_stats_btn = gr.Button("\u5237\u65b0\u7edf\u8ba1\u4fe1\u606f", variant="secondary")
                paper_list_display = gr.Markdown("\u77e5\u8bc6\u5e93\u4e3a\u7a7a\u3002")
                stats_display = gr.Markdown("")
                pdf_upload.upload(upload_pdf, inputs=[pdf_upload], outputs=[upload_status])
                delete_btn.click(delete_paper, inputs=[delete_input], outputs=[delete_status]).then(refresh_papers, outputs=[paper_list_display])
                refresh_list_btn.click(refresh_papers, outputs=[paper_list_display])
                refresh_stats_btn.click(refresh_stats, outputs=[stats_display])
        
            with gr.TabItem("论文分析"):
                with gr.Row():
                    paper_select = gr.Dropdown(label="选择论文", choices=[], interactive=True, scale=2)
                    analysis_provider = gr.Dropdown(label="LLM厂商", choices=["\u9ed8\u8ba4"], value="\u9ed8\u8ba4", interactive=True, scale=1)
                with gr.Row():
                    method_btn = gr.Button("🔍 一键解析研究方法", variant="primary", scale=1)
                    flowchart_btn = gr.Button("📊 一键分析算法流程", variant="secondary", scale=1)
                method_output = gr.Markdown(label="研究方法分析结果", value="", elem_classes="paper-box")
                with gr.Row() as flowchart_row:
                    with gr.Column():
                        gr.Markdown("### 算法步骤说明")
                with gr.Row() as flowchart_row:
                    flowchart_output = gr.Markdown(value="", label="算法流程分析结果", elem_classes="paper-box")
                def update_paper_dropdown():
                    try:
                        import httpx
                        r = httpx.get("http://127.0.0.1:8000/api/papers", timeout=10)
                        papers = r.json()
                        return gr.Dropdown(choices=[p["file_name"] for p in papers])
                    except:
                        return gr.Dropdown(choices=[])

                method_btn.click(analyze_paper_method, inputs=[paper_select, analysis_provider], outputs=[method_output])
                flowchart_btn.click(generate_paper_flowchart, inputs=[paper_select, analysis_provider], outputs=[flowchart_output])
        app.load(update_paper_dropdown, outputs=[paper_select])
        app.load(lambda: _get_paper_choices(), outputs=[compare_paper_1])
        app.load(lambda: _get_paper_choices(), outputs=[compare_paper_2])
        app.load(refresh_papers, outputs=[paper_list_display])
        app.load(lambda: gr.Dropdown(choices=["\u5168\u90e8"] + [p["file_name"] for p in __import__("httpx").get("http://127.0.0.1:8000/api/papers", timeout=10).json()]), outputs=[paper_filter])
        app.load(lambda: gr.Dropdown(choices=["\u5168\u90e8"]), outputs=[paper_filter])
    return app


def run_gradio():
    cfg = get_config()
    app = build_app()
    app.launch(server_name=cfg.frontend_host, server_port=cfg.frontend_port, share=cfg.frontend_share)


if __name__ == "__main__":
    run_gradio()