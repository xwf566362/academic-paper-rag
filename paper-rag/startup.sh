 #!/bin/bash
# 论文RAG知识库 - Linux/Mac 启动脚本
 
 set -e
 
 echo "============================================"
 echo "   论文RAG知识库 - 启动脚本"
 echo "   基于本地BGE嵌入 + 远程LLM API"
 echo "============================================"
 echo ""
 
# 检查Python
 if ! command -v python3 &> /dev/null; then
     echo "[错误] 未找到 python3"
     exit 1
 fi
 
# 检查依赖
 echo "[1/3] 检查依赖..."
 python3 -c "import fitz, chromadb, sentence_transformers, fastapi, gradio, httpx, yaml" 2>/dev/null
 if [ $? -ne 0 ]; then
     echo "[提示] 部分依赖未安装，执行: pip install -r requirements.txt"
     exit 1
 fi
 
# 检查配置
 echo "[2/3] 检查配置..."
 if [ ! -f config.yaml ]; then
     echo "[错误] config.yaml 未找到"
     exit 1
 fi
 
# 启动
 echo "[3/3] 启动服务..."
 echo ""
 echo "FastAPI 后端: http://127.0.0.1:8000"
 echo "Gradio 前端: http://127.0.0.1:7860"
 echo ""
 echo "请在 config.yaml 中配置有效的 API 密钥"
 echo ""
 
 python3 run.py --mode all
