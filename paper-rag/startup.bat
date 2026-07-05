 @echo off
 chcp 65001 >nul
 title 论文RAG知识库
 
 echo ============================================
 echo    论文RAG知识库 - 启动脚本
 echo    基于本地BGE嵌入 + 远程LLM API
 echo ============================================
 echo.
 
 :: 检查Python
 where python >nul 2>&1
 if %errorlevel% neq 0 (
     echo [错误] 未找到 Python，请确保已安装 Python 3.9+
     pause
     exit /b 1
 )
 
 :: 检查依赖
 echo [1/3] 检查依赖...
 python -c "import fitz, chromadb, sentence_transformers, fastapi, gradio, httpx, yaml" 2>nul
 if %errorlevel% neq 0 (
     echo [提示] 部分依赖未安装，执行: pip install -r requirements.txt
     pause
     exit /b 1
 )
 
 :: 检查配置文件
 echo [2/3] 检查配置...
 if not exist config.yaml (
     echo [错误] config.yaml 未找到，请确保配置文件存在
     pause
     exit /b 1
 )
 
 :: 启动服务
 echo [3/3] 启动服务...
 echo.
 echo FastAPI 后端: http://127.0.0.1:8000
 echo Gradio 前端: http://127.0.0.1:7860
 echo.
 echo 请在 config.yaml 中配置有效的 API 密钥
 echo.
 
 python run.py --mode all
 
 pause
