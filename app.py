import os
import sys
import atexit
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from ui.app import build_app, ensure_pipeline, get_pipeline, CSS, APP_THEME


if __name__ == "__main__":
    ensure_pipeline()
    atexit.register(lambda: get_pipeline() and get_pipeline().close())
    print("[App] 系统初始化完成，启动 Web 界面...")
    app = build_app()
    webbrowser.open("http://127.0.0.1:7861")
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True,
        css=CSS,
        theme=APP_THEME,
    )
