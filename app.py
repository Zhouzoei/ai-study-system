import os
import sys
import atexit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from ui.app import ensure_pipeline, get_pipeline


if __name__ == "__main__":
    pipeline = ensure_pipeline()
    pipeline.start_background_agent()
    atexit.register(lambda: get_pipeline() and get_pipeline().background_agent.stop())
    atexit.register(lambda: get_pipeline() and get_pipeline().close())
    print("[App] 系统初始化完成，后台 Agent 已启动，启动 Streamlit 界面...")
    print("[App] 访问 http://127.0.0.1:7861")

    os.system("streamlit run ui/app.py --server.port 7861 --server.address 127.0.0.1 --server.headless true")
