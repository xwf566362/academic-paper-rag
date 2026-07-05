"""统一启动入口"""

import logging
import subprocess
import sys
import time
import httpx

from app.config import get_config, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s")
logger = logging.getLogger("run")


def check_api_key():
    cfg = get_config()
    provider = cfg.default_provider
    key = cfg.llm_providers[provider].api_key
    if not key or key.startswith("your-"):
        print("\n" + "=" * 55)
        print("  !!! API 密钥未配置 !!!")
        print(f"  默认厂商: {provider}")
        print("  请在 config.yaml 中填入有效的 api_key")
        print("=" * 55 + "\n")
        return False
    return True


def main():
    ensure_dirs()
    cfg = get_config()
    check_api_key()

    if len(sys.argv) > 1 and sys.argv[1] == "backend":
        import uvicorn
        uvicorn.run("app.main:app", host=cfg.server_host, port=cfg.server_port, reload=False, log_level="info")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "frontend":
        from app.frontend.gradio_app import run_gradio
        run_gradio()
        return

    # 默认模式：子进程启动后端，主进程启动前端
    print("\n" + "=" * 45)
    print("  论文RAG知识库启动中...")
    print(f"  后端: http://127.0.0.1:{cfg.server_port}")
    print(f"  前端: http://127.0.0.1:{cfg.frontend_port}")
    print("=" * 45 + "\n")

    proc = subprocess.Popen(
        [sys.executable, __file__, "backend"],
        stdout=None, stderr=None,
    )
    time.sleep(2)

    # 等待后端就绪
    for i in range(30):
        try:
            r = httpx.get(f"http://127.0.0.1:{cfg.server_port}/health", timeout=2)
            if r.status_code == 200:
                logger.info("后端就绪，启动前端...")
                break
        except Exception:
            pass
        if i == 0:
            logger.info("等待后端启动...")
        time.sleep(1)
    else:
        logger.error("后端启动超时，请单独运行 python run.py backend 查看错误")
        return

    from app.frontend.gradio_app import run_gradio
    try:
        run_gradio()
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()