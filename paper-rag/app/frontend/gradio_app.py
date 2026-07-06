"""Gradio 前端界面"""
import logging
import os
from pathlib import Path
from typing import Tuple
import json
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


async def ask_question(question: str, paper_filter: str, history: list, provider: str = "") -> Tuple[list, str]:
    if not question.strip():
        return history, ""
    req = {"question": question}
    if provider:
        req["provider"] = provider
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


async def load_providers() -> list:
    """Fetch available providers from the backend."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{API_BASE}/config/providers")
            data = resp.json()
            return [(
                f"{name} ({p["model"]})" if not p["configured"] else f"{name} ({p["model"]}) - configured",
                name,
            ) for name, p in data.get("providers", {}).items()]
    except Exception:
        return [("(no providers)", "")]


async def save_api_config(provider: str, api_key: str) -> str:
    """Save API provider configuration to backend."""
    if not provider or not api_key:
        return "Please select a provider and enter an API key."
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{API_BASE}/config/provider",
                json={"provider": provider, "api_key": api_key},
            )
        if resp.status_code == 200:
            return f"OK - configured {provider}"
        else:
            return f"Error: {resp.text}"
    except Exception as e:
        return f"Connection failed: {e}"


async def upload_papers_batch(files):
    """Upload multiple PDFs via batch endpoint."""
    if not files:
        return "No files selected."
    import httpx
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            data = []
            for f in files:
                data.append(("files", (f.name, open(f, "rb"), "application/pdf")))
            resp = await client.post(f"{API_BASE}/upload/batch", files=data)
        result = resp.json()
        statuses = []
        for r in result.get("results", []):
            icon = "\u2705" if r["status"] == "ok" else "\u274c"
            statuses.append(f"{icon} {r['file']}: {r.get('chunks', 'error')} chunks")
        return "\n".join(statuses) if statuses else "No files processed."
    except Exception as e:
        return f"Batch upload failed: {e}"

async def display_paper_detail(paper_name):
    """Fetch and display paper metadata (summary, keywords)."""
    if not paper_name:
        return "### No paper selected", "", ""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{API_BASE}/papers/{paper_name}/meta")
        meta = resp.json()
        title = meta.get("title", paper_name)
        summary = meta.get("summary", "")
        keywords = meta.get("keywords", [])
        kw_text = ", ".join(keywords) if keywords else "No keywords"
        return f"### {title}", summary or "*No summary generated*", f"**Keywords:** {kw_text}"
    except Exception as e:
        return f"### {paper_name}", f"Error: {e}", ""

async def generate_summary_action(paper_name):
    """Generate summary + keywords via LLM."""
    if not paper_name:
        return "Select a paper first."
    import httpx
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{API_BASE}/papers/{paper_name}/summary")
        result = resp.json()
        return f"Summary generated ({len(result.get('keywords', []))} keywords)"
    except Exception as e:
        return f"Error: {e}"

async def export_bibtex_action(paper_name):
    """Export BibTeX citation."""
    if not paper_name:
        return "Select a paper first."
    import httpx
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(f"{API_BASE}/papers/{paper_name}/bibtex")
        if resp.status_code != 200:
            return f"Export failed: {resp.json().get('detail', 'Unknown')}"
        result = resp.json()
        bibtex = result.get("bibtex", "")
        name_clean = paper_name.replace('.pdf', '').replace('.PDF', '')
        bib_path = os.path.join(os.path.expanduser("~"), "Desktop", f"{name_clean}.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(bibtex)
        return f"Saved to {bib_path} ({len(bibtex)} chars)"
    except Exception as e:
        return f"Error: {e}"

async def refresh_paper_selector():
    """Update the paper selector dropdown in the management tab."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{API_BASE}/papers")
        papers = resp.json()
        return gr.Dropdown(choices=[p["file_name"] for p in papers])
    except Exception:
        return gr.Dropdown(choices=[])


async def load_current_config():
    """Load current API provider config from backend."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{API_BASE}/config/providers")
        data = resp.json()
        default = data.get("default", "")
        providers = data.get("providers", {})
        lines = []
        for name, info in providers.items():
            icon = chr(9989) if info.get("configured") else chr(10060)
            lines.append(f"{icon} {name} ({info.get('model', '')})")
        status = "**Current config:**\n" + "\n".join(lines)
        return default, "", status
    except Exception as e:
        return "tongyi", "", f"Error: {e}"


CSS = """
body .gradio-container { max-width: 100% !important; padding: 10px 20px !important; }
body { background: #f5f7fa !important; }
.app-header { background: linear-gradient(135deg, #0284c7 0%, #7dd3fc 100%); color: white; padding: 18px 24px; border-radius: 10px; margin-bottom: 16px; box-shadow: 0 4px 15px rgba(2,132,199,0.12); }
.app-header h1 { margin: 0; font-size: 22px; font-weight: 700; }
.app-header p { margin: 10px 0 0; opacity: 0.88; font-size: 14px; letter-spacing: 0.3px; }
.tab-nav { background: white !important; border-radius: 10px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important; }
.tab-nav button { border-radius: 0 !important; padding: 14px 22px !important; font-size: 14px !important; font-weight: 500 !important; color: #4a5568 !important; border-bottom: 2px solid transparent !important; }
.tab-nav button[aria-selected="true"] { color: #0f172a !important; border-bottom-color: #94a3b8 !important; background: #f1f5f9 !important; }
.paper-box { background: white; border: 1px solid #cbd5e1; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
button.gr-button { border-radius: 8px !important; font-weight: 500 !important; }
input, textarea, select { border-radius: 8px !important; border: 1px solid #cbd5e1 !important; }
.gr-chatbot { border-radius: 10px !important; border: 1px solid #cbd5e1 !important; }
div[data-testid="block"] { border-radius: 10px !important; background: white !important; padding: 12px !important; box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important; }
footer { display: none !important; }
"""

def build_app() -> gr.Blocks:
    cfg = get_config()
    providers = ["\u9ed8\u8ba4"] + list(cfg.llm_providers.keys())
    with gr.Blocks(title="PaperMind \u00b7 \u5b66\u672f\u667a\u6790", css=CSS) as app:
        gr.HTML("""<div class=\"app-header\"><h1>PaperMind \u00b7 \u5b66\u672f\u667a\u6790</h1><p>\u4e0a\u4f20\u8bba\u6587 \u2192 AI \u81ea\u52a8\u89e3\u6790 \u2192 \u667a\u80fd\u95ee\u7b54 \u00b7 \u6458\u8981\u751f\u6210 \u00b7 \u6587\u732e\u5bfc\u51fa</p></div>""")
        with gr.Tabs():
            with gr.TabItem("\u2699\ufe0f \u914d\u7f6e"):
                gr.Markdown("### Configure LLM Provider")
                with gr.Row():
                    provider_dd = gr.Dropdown(label="Provider", choices=["tongyi","deepseek","openai"], value="tongyi", scale=1)
                    api_key_input = gr.Textbox(label="API Key", type="password", placeholder="sk-...", scale=2)
                save_btn = gr.Button("Save", variant="primary")
                config_status = gr.Markdown("Enter API key and click Save.")
                save_btn.click(save_api_config, inputs=[provider_dd, api_key_input], outputs=[config_status])
            with gr.TabItem("\u8bba\u6587\u95ee\u7b54"):
                chatbot = gr.Chatbot(label="\u5bf9\u8bdd\u8bb0\u5f55", height=500)
                with gr.Row():
                    msg = gr.Textbox(label="\u8f93\u5165\u95ee\u9898", placeholder="\u8bf7\u8f93\u5165\u5173\u4e8e\u8bba\u6587\u7684\u95ee\u9898...", scale=4)
                    submit_btn = gr.Button("\u53d1\u9001", variant="primary", scale=1)
                with gr.Row():
                    paper_filter = gr.Dropdown(label="\u9650\u5b9a\u8bba\u6587", choices=["\u5168\u90e8"], value="\u5168\u90e8", interactive=True, scale=2)
                    clear_btn = gr.Button("\u6e05\u7a7a\u5bf9\u8bdd", size="sm", scale=1)
                submit_btn.click(ask_question, inputs=[msg, paper_filter, chatbot, provider_dd], outputs=[chatbot, msg])
                msg.submit(ask_question, inputs=[msg, paper_filter, chatbot, provider_dd], outputs=[chatbot, msg])
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
                    with gr.Column(scale=2):
                        gr.Markdown("### \u4e0a\u4f20\u8bba\u6587")
                        pdf_upload = gr.File(label="\u9009\u62e9PDF\u6587\u4ef6", file_types=[".pdf"], file_count="multiple")
                        upload_btn = gr.Button("\u6279\u91cf\u4e0a\u4f20", variant="primary")
                        upload_status = gr.Markdown("")
                        refresh_list_btn = gr.Button("\u5237\u65b0\u5217\u8868", variant="secondary")
                    with gr.Column(scale=1):
                        gr.Markdown("### \u8bba\u6587\u8be6\u60c5")
                        paper_selector = gr.Dropdown(label="\u9009\u62e9\u8bba\u6587", choices=[], interactive=True)
                        paper_title = gr.Markdown("### \u672a\u9009\u62e9\u8bba\u6587")
                        paper_summary = gr.Markdown("")
                        paper_keywords = gr.Markdown("")
                        with gr.Row():
                            summary_btn = gr.Button("\u751f\u6210\u6458\u8981", size="sm")
                            bibtex_btn = gr.Button("\u5bfc\u51fa\u53c2\u8003\u6587\u732e", size="sm")
                        action_status = gr.Markdown("")
                gr.Markdown("---")
                with gr.Row():
                    delete_input = gr.Textbox(label="\u5220\u9664\u8bba\u6587", placeholder="\u6587\u4ef6\u540d.pdf", scale=3)
                    delete_btn = gr.Button("\u5220\u9664", variant="stop", scale=1)
                    delete_status = gr.Markdown("")
                paper_list_display = gr.Markdown("\u77e5\u8bc6\u5e93\u4e3a\u7a7a\u3002")
                stats_display = gr.Markdown("")
                upload_btn.click(upload_papers_batch, inputs=[pdf_upload], outputs=[upload_status]).then(refresh_paper_selector, outputs=[paper_selector]).then(lambda: (_get_paper_choices(), _get_paper_choices()), outputs=[compare_paper_1, compare_paper_2])
                summary_btn.click(generate_summary_action, inputs=[paper_selector], outputs=[action_status]).then(display_paper_detail, inputs=[paper_selector], outputs=[paper_title, paper_summary, paper_keywords])
                bibtex_btn.click(export_bibtex_action, inputs=[paper_selector], outputs=[action_status])
                paper_selector.change(display_paper_detail, inputs=[paper_selector], outputs=[paper_title, paper_summary, paper_keywords])
                delete_btn.click(delete_paper, inputs=[delete_input], outputs=[delete_status]).then(refresh_papers, outputs=[paper_list_display]).then(refresh_paper_selector, outputs=[paper_selector]).then(lambda: (_get_paper_choices(), _get_paper_choices()), outputs=[compare_paper_1, compare_paper_2])
                refresh_list_btn.click(refresh_papers, outputs=[paper_list_display])
            with gr.TabItem("论文分析"):
                with gr.Row():
                    paper_select = gr.Dropdown(label="选择论文", choices=[], interactive=True, scale=2)
                    analysis_provider = gr.Dropdown(label="LLM厂商", choices=["\u9ed8\u8ba4"], value="\u9ed8\u8ba4", interactive=True, scale=1)
                with gr.Row():
                    method_btn = gr.Button("🔍 一键解析研究方法", variant="primary", scale=1)
                    flowchart_btn = gr.Button("📊 一键分析算法流程", variant="secondary", scale=1)
                    clear_analysis_btn = gr.Button("清空", size="sm", scale=1)
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
                clear_analysis_btn.click(lambda: ("", ""), outputs=[method_output, flowchart_output])
        app.load(update_paper_dropdown, outputs=[paper_select])
        app.load(lambda: _get_paper_choices(), outputs=[compare_paper_1])
        app.load(lambda: _get_paper_choices(), outputs=[compare_paper_2])
        app.load(refresh_papers, outputs=[paper_list_display])
        app.load(refresh_paper_selector, outputs=[paper_selector])
        app.load(lambda: gr.Dropdown(choices=["\u5168\u90e8"] + [p["file_name"] for p in __import__("httpx").get("http://127.0.0.1:8000/api/papers", timeout=10).json()]), outputs=[paper_filter])
        app.load(lambda: gr.Dropdown(choices=["\u5168\u90e8"]), outputs=[paper_filter])

        app.load(load_current_config, outputs=[provider_dd, api_key_input, config_status])
    return app


def run_gradio():
    cfg = get_config()
    app = build_app()
    app.launch(server_name=cfg.frontend_host, server_port=cfg.frontend_port, share=cfg.frontend_share)


if __name__ == "__main__":
    run_gradio()