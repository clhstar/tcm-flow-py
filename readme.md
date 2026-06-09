每次打开项目终端后，先执行：

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 启动项目
uvicorn app.main:app --reload --port 2026