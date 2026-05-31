import os
import sys
import json
import time
import uuid
import logging
from typing import Optional

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

import streamlit as st

from engines.pipeline import EnhancedRAGPipeline
from utils.llm_service import LLMService
from utils.embedding_service import EmbeddingService

st.set_page_config(
    page_title="AI 学习系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──

st.markdown("""
<style>
    .main > div { padding-bottom: 2rem; }
    .stChatMessage { border-radius: 12px; padding: 8px 12px; }
    .stChatMessage[data-testid="user-message"] { background: #eef2ff; }
    .stChatMessage[data-testid="assistant-message"] { background: #f8fafc; }
    .stProgress > div > div > div { background: linear-gradient(90deg, #6366f1, #22c55e); }
    div[data-testid="stSidebarNav"] { display: none; }
    h1 { font-size: 1.8rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.3rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.1rem !important; font-weight: 500 !important; }
    .st-emotion-cache-1y4p8pa { padding: 2rem 1rem; }
    .st-emotion-cache-18ni7ap { font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #4338ca; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; font-weight: 500; }
    .quiz-card {
        background: #f8fafc; border-radius: 12px; padding: 20px;
        border-left: 4px solid #6366f1; margin: 12px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .kg-card {
        background: #f8fafc; border-radius: 12px; padding: 16px;
        border-left: 4px solid #6366f1; margin: 8px 0;
    }
    .progress-card {
        background: #f0fdf4; border-radius: 12px; padding: 16px;
        border-left: 4px solid #22c55e; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ──

def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess_{uuid.uuid4().hex[:12]}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False
    if "pipeline_error" not in st.session_state:
        st.session_state.pipeline_error = ""
    if "current_doc_id" not in st.session_state:
        st.session_state.current_doc_id = None
    if "quiz_history" not in st.session_state:
        st.session_state.quiz_history = []
    if "init_in_progress" not in st.session_state:
        st.session_state.init_in_progress = False

init_session_state()


# ── Pipeline initialization ──

_embed_service: Optional[EmbeddingService] = None


def get_pipeline():
    return st.session_state.get("_pipeline")


def ensure_pipeline():
    global _embed_service
    existing = get_pipeline()
    if existing is not None:
        return existing
    _embed_service = EmbeddingService()
    llm_service = LLMService()
    pipeline = EnhancedRAGPipeline(
        embed_func=_embed_service.embed,
        llm_func=llm_service.invoke,
        llm_service=llm_service,
    )
    st.session_state["_pipeline"] = pipeline
    return pipeline


def get_stats():
    pipeline = get_pipeline()
    if not pipeline:
        return {}
    try:
        return pipeline.get_stats()
    except Exception:
        return {}


# ── Utility Functions ──

def parse_quiz_output(raw: str, sub_type: str) -> dict:
    """Parse LLM quiz output into structured data."""
    import re as _re
    result = {"question": "", "options": [], "correct_answer": "", "explanation": "", "sub_type": sub_type}

    qm = _re.search(r"##\s*题目\s*\n(.+?)(?=\n##\s*选项|\n##\s*答案|\n##\s*正确答案|\n##\s*参考答案)", raw, _re.DOTALL)
    if qm:
        result["question"] = qm.group(1).strip()

    if sub_type == "choice":
        om = _re.search(r"##\s*选项\s*\n(.+?)(?=\n##\s*答案|\n##\s*正确答案|\n##\s*解析|\n##\s*参考答案)", raw, _re.DOTALL)
        if om:
            opts = _re.findall(r'([A-D])[.、]\s*(.+?)(?=\n[A-D][.、]|$)', om.group(1).strip(), _re.DOTALL)
            result["options"] = [{"label": m[0], "text": m[1].strip()} for m in opts]

    am = _re.search(r"##\s*(?:正确答案|答案)\s*\n(.+?)(?=\n##\s*解析|\n##\s*评分要点|\n##\s*风格说明|$)", raw, _re.DOTALL)
    if am:
        result["correct_answer"] = am.group(1).strip().rstrip(".")

    em = _re.search(r"##\s*解析\s*\n(.+?)(?=\n##|$)", raw, _re.DOTALL)
    if em:
        result["explanation"] = em.group(1).strip()

    if sub_type == "judgment" and not result["correct_answer"]:
        if "✅" in raw or "正确" in raw:
            result["correct_answer"] = "正确"
        elif "❌" in raw or "错误" in raw:
            result["correct_answer"] = "错误"

    if not result["options"] and sub_type == "choice":
        fallback = _re.findall(r'([A-D])[.、]\s*(.+?)(?=\n[A-D][.、]|\n##|\n\*\*|$)', raw, _re.DOTALL)
        if fallback:
            result["options"] = [{"label": m[0], "text": m[1].strip()[:120]} for m in fallback]

    return result


def get_entity_info(entity_name: str) -> str:
    pipeline = get_pipeline()
    if not pipeline:
        return "系统未初始化"
    try:
        result = pipeline.knowledge_graph.query_entity(entity_name)
        if not result:
            candidates = pipeline.knowledge_graph.search_entities(entity_name, limit=5)
            if candidates:
                result = pipeline.knowledge_graph.query_entity(candidates[0]["name"])
        if not result:
            return f"未找到实体「{entity_name}」"
        entity = result.get("entity", result) if isinstance(result, dict) else result
        name = entity.get("name", entity_name)
        etype = entity.get("entity_type", entity.get("type", ""))
        props = entity.get("properties", {})
        parts = [f"**{name}**"]
        if etype:
            parts.append(f"类型: {etype}")
        if props:
            for k, v in list(props.items())[:6]:
                parts.append(f"- {k}: {str(v)[:200]}")
        relations = result.get("relations", []) if isinstance(result, dict) else []
        if relations:
            parts.append("\n**关联:**")
            for r in relations[:10]:
                if isinstance(r, dict):
                    parts.append(f"- {r.get('source_name','?')} → {r.get('target_name','?')} ({r.get('type','')})")
        return "\n".join(parts)
    except Exception as e:
        return f"查询失败: {e}"


def ensure_session():
    pipeline = get_pipeline()
    if not pipeline:
        return
    session_id = st.session_state.session_id
    try:
        session = pipeline.conversation_memory.get_session(session_id)
        if not session:
            pipeline.conversation_memory.create_session(user_id="default", title="Streamlit Session")
    except Exception as e:
        logger.warning(f"Session init failed for {session_id}: {e}")


def handle_dispatch(message: str) -> str:
    pipeline = get_pipeline()
    if not pipeline:
        return "系统未初始化"
    ensure_session()
    session_id = st.session_state.session_id
    try:
        result = pipeline.dispatch(
            message=message,
            session_id=session_id,
            doc_id=st.session_state.current_doc_id,
        )
        return result.get("answer", "处理失败")
    except Exception as e:
        return f"处理失败: {e}"


# ── Sidebar ──

with st.sidebar:
    st.markdown("### 🧠 AI 学习系统")
    st.markdown("---")

    if st.session_state.pipeline_ready:
        stats = get_stats()
        storage = stats.get("storage", {})
        conv = stats.get("conversation", {})

        col1, col2 = st.columns(2)
        with col1:
            st.metric("文档", storage.get("doc_count", 0))
            st.metric("节点", storage.get("total_nodes", 0))
        with col2:
            st.metric("对话", conv.get("total_sessions", 0))
            st.metric("消息", conv.get("total_messages", 0))

        pipeline = get_pipeline()
        if pipeline and hasattr(pipeline, 'learning_loop'):
            try:
                lstate = pipeline.learning_loop.get_learner_state(user_id="default")
                progress = lstate.get("progress", {})
                pct = progress.get("progress_pct", 0)
                st.markdown("---")
                st.markdown(f"### 📊 掌握度 **{pct}%**")
                st.progress(pct / 100.0)
                if lstate.get("has_weak_nodes"):
                    st.warning(f"⚠️ {len(lstate['weak_nodes'])} 个薄弱点")
                if lstate.get("has_due_reviews"):
                    st.info(f"📅 {len(lstate['due_reviews'])} 个待复习")
            except Exception:
                pass

        st.markdown("---")
        st.markdown("### 📋 导航")
        selected_page = st.radio(
            "选择功能",
            ["💬 对话", "📄 文档管理", "🧪 出题", "📊 学习进度", "🔗 知识图谱"],
            label_visibility="collapsed",
            index=0,
        )
        st.session_state.selected_page = selected_page

        st.markdown("---")
        if st.button("🔄 新建会话", width="stretch"):
            st.session_state.session_id = f"sess_{uuid.uuid4().hex[:12]}"
            st.session_state.messages = []
            st.rerun()

        if st.button("🗑️ 清空对话", width="stretch"):
            st.session_state.messages = []
            st.rerun()


# ── Pages ──

def render_chat_page():
    st.markdown("## 💬 智能问答")
    st.markdown("上传文档后，我可以帮你 answering 问题、总结内容、出题测试。")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        with chat_container:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_answer = ""
                try:
                    pipeline = get_pipeline()
                    ensure_session()
                    if pipeline and hasattr(pipeline, 'process_with_loop'):
                        for event in pipeline.process_with_loop(
                            message=prompt,
                            session_id=st.session_state.session_id,
                            doc_id=st.session_state.current_doc_id,
                        ):
                            et = event.get("type")
                            if et == "token" or et == "answer_chunk":
                                full_answer += event.get("content", "")
                                placeholder.markdown(full_answer + "▌")
                            elif et == "full" or et == "answer":
                                full_answer = event.get("content", full_answer)
                                placeholder.markdown(full_answer)
                            elif et == "progress":
                                placeholder.markdown("⏳ " + event.get("content", ""))
                            elif et == "degraded":
                                full_answer = event.get("content", full_answer)
                                placeholder.markdown(full_answer)
                            elif et == "error":
                                full_answer = event.get("content", "处理失败")
                                placeholder.markdown(f"❌ {full_answer}")
                            elif et == "reflection_token":
                                full_answer += event.get("content", "")
                                placeholder.markdown(full_answer)
                            elif et == "loop_answer":
                                full_answer = event.get("content", full_answer)
                                placeholder.markdown(full_answer)
                    else:
                        full_answer = handle_dispatch(prompt)
                        placeholder.markdown(full_answer)
                except Exception as e:
                    placeholder.error(f"处理失败: {e}")
                st.session_state.messages.append({"role": "assistant", "content": full_answer})

    quick_actions = st.container()
    with quick_actions:
        st.markdown("---")
        cols = st.columns(5)
        qa_texts = ["什么是过拟合", "总结一下文档", "出个选择题", "复习薄弱点", "对比概念"]
        for i, qt in enumerate(qa_texts):
            if cols[i].button(qt, width="stretch"):
                st.session_state.messages.append({"role": "user", "content": qt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(qt)
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        full_answer = ""
                        try:
                            pipeline = get_pipeline()
                            ensure_session()
                            if pipeline and hasattr(pipeline, 'process_with_loop'):
                                for event in pipeline.process_with_loop(
                                    message=qt,
                                    session_id=st.session_state.session_id,
                                    doc_id=st.session_state.current_doc_id,
                                ):
                                    et = event.get("type")
                                    if et == "token" or et == "answer_chunk":
                                        full_answer += event.get("content", "")
                                        placeholder.markdown(full_answer + "▌")
                                    elif et == "full" or et == "answer":
                                        full_answer = event.get("content", full_answer)
                                        placeholder.markdown(full_answer)
                                    elif et == "progress":
                                        placeholder.markdown("⏳ " + event.get("content", ""))
                                    elif et == "degraded":
                                        full_answer = event.get("content", full_answer)
                                        placeholder.markdown(full_answer)
                                    elif et == "error":
                                        full_answer = event.get("content", "处理失败")
                                        placeholder.markdown(f"❌ {full_answer}")
                                    elif et == "reflection_token":
                                        full_answer += event.get("content", "")
                                        placeholder.markdown(full_answer)
                                    elif et == "loop_answer":
                                        full_answer = event.get("content", full_answer)
                                        placeholder.markdown(full_answer)
                                placeholder.markdown(full_answer)
                            else:
                                full_answer = handle_dispatch(qt)
                                placeholder.markdown(full_answer)
                        except Exception:
                            full_answer = handle_dispatch(qt)
                            placeholder.markdown(full_answer)
                        st.session_state.messages.append({"role": "assistant", "content": full_answer})


def render_documents_page():
    st.markdown("## 📄 文档管理")
    pipeline = get_pipeline()
    if not pipeline:
        st.error("系统未初始化，请稍后重试")
        return

    with st.expander("📤 上传文档", expanded=True):
        uploaded_file = st.file_uploader(
            "选择文件 (PDF/TXT/MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=False,
        )
        if uploaded_file is not None:
            MAX_UPLOAD_SIZE = 50 * 1024 * 1024
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > MAX_UPLOAD_SIZE:
                st.error(f"文件过大（{len(file_bytes) / 1024 / 1024:.1f}MB），最大支持 50MB")
            else:
                import tempfile
                from pathlib import Path

                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                with st.spinner("正在处理文档..."):
                    try:
                        result = pipeline.document_manager.ingest_document(
                            file_path=tmp_path,
                            source_name=uploaded_file.name,
                        )
                        doc_id = result.get("doc_id", "")
                        st.session_state.current_doc_id = doc_id
                        st.success(f"✅ 文档「{uploaded_file.name}」处理完成！")
                        if "stats" in result:
                            st.info(f"提取了 {result['stats'].get('total_nodes', 0)} 个知识节点")
                    except Exception as e:
                        st.error(f"处理失败: {e}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

    st.markdown("### 📚 已上传文档")
    try:
        docs = pipeline.document_manager.list_documents()
        if docs:
            for doc in docs:
                doc_id = doc.get("doc_id", "")
                title = doc.get("title", doc.get("source_name", "未命名"))
                status = doc.get("status", "unknown")
                created = doc.get("created_at", "")
                status_icon = {"ready": "✅", "processing": "⏳", "error": "❌"}.get(status, "❓")
                with st.container():
                    cols = st.columns([3, 1, 1, 1])
                    cols[0].markdown(f"**{title}**")
                    cols[1].markdown(f"{status_icon} {status}")
                    cols[2].markdown(f"ID: {doc_id[:8]}...")
                    if cols[3].button("选择", key=f"sel_{doc_id}"):
                        st.session_state.current_doc_id = doc_id
                        st.success(f"已选择文档: {title}")
        else:
            st.info("暂无上传的文档。上传一份 PDF/TXT/MD 文档开始学习。")
    except Exception as e:
        st.info("文档管理器尚未初始化或暂无文档。")


def render_quiz_page():
    st.markdown("## 🧪 出题测试")
    pipeline = get_pipeline()
    if not pipeline:
        st.error("系统未初始化")
        return

    # Init quiz state
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    if "quiz_history" not in st.session_state:
        st.session_state.quiz_history = []

    tab1, tab2, tab3 = st.tabs(["📝 普通出题", "🎯 自适应出题", "🎨 风格模仿"])

    # ── Helper: run quiz generation and parse ──
    def _generate_and_parse(result, sub_type: str):
        if not result or not result.answer:
            st.error("生成失败，请重试")
            return
        parsed = parse_quiz_output(result.answer, sub_type)
        if not parsed.get("question"):
            st.warning("题目解析失败，显示原始输出")
            st.markdown(result.answer)
            return
        parsed["knowledge_node_ids"] = result.metadata.get("knowledge_node_ids", [])
        parsed["raw"] = result.answer
        parsed["answered"] = False
        parsed["is_correct"] = None
        parsed["user_answer"] = None
        st.session_state.quiz_data = parsed
        st.rerun()

    # ── Tab 1: Normal Quiz ──
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            topic = st.text_input("出题主题", placeholder="输入主题，留空则随机出题", key="quiz_topic")
        with col2:
            q_type = st.selectbox("题型", ["choice", "judgment", "fill", "essay"],
                                  format_func=lambda x: {"choice": "选择题", "judgment": "判断题",
                                                         "fill": "填空题", "essay": "简答题"}[x],
                                  key="quiz_type")
        if st.button("🎲 生成题目", key="gen_quiz", width="stretch"):
            with st.spinner("正在生成题目..."):
                try:
                    result = pipeline.quiz_agent.generate(
                        message=topic or "请出题", topic=topic,
                        sub_type=q_type, doc_id=st.session_state.current_doc_id,
                    )
                    _generate_and_parse(result, q_type)
                except Exception as e:
                    st.error(f"出题失败: {e}")

    # ── Tab 2: Adaptive Quiz ──
    with tab2:
        st.markdown("根据你的薄弱知识点自动生成针对性题目。")
        aq_type = st.selectbox("题型", ["choice", "judgment", "fill", "essay"],
                               format_func=lambda x: {"choice": "选择题", "judgment": "判断题",
                                                      "fill": "填空题", "essay": "简答题"}[x],
                               key="aq_type")
        if st.button("🎯 生成自适应题目", width="stretch", type="primary", key="gen_adaptive"):
            with st.spinner("正在分析薄弱点并出题..."):
                try:
                    result = pipeline.quiz_agent.generate_adaptive_quiz(
                        user_id="default", sub_type=aq_type,
                    )
                    if result.metadata.get("weak_count", 0) == 0:
                        st.info("🎉 暂无薄弱知识点！你可以用普通出题随机练习。")
                    else:
                        _generate_and_parse(result, aq_type)
                except Exception as e:
                    st.error(f"生成失败: {e}")

    # ── Tab 3: Styled Quiz ──
    with tab3:
        st.markdown("模仿文档的出题风格生成题目。")
        sq_type = st.selectbox("题型", ["choice", "judgment", "fill", "essay"],
                               format_func=lambda x: {"choice": "选择题", "judgment": "判断题",
                                                      "fill": "填空题", "essay": "简答题"}[x],
                               key="sq_type")
        if st.button("🎨 生成风格化题目", width="stretch", type="primary", key="gen_styled"):
            with st.spinner("正在分析文档风格并出题..."):
                try:
                    result = pipeline.quiz_agent.generate_styled_quiz(
                        topic="", sub_type=sq_type, doc_id=st.session_state.current_doc_id,
                    )
                    _generate_and_parse(result, sq_type)
                except Exception as e:
                    st.error(f"生成失败: {e}")

    # ── Interactive Answer Area (shown when quiz_data exists) ──
    qd = st.session_state.quiz_data
    if qd and qd.get("question"):
        st.markdown("---")
        sub_type = qd.get("sub_type", "choice")

        # Question display
        st.markdown(f"""
        <div class="quiz-card">
            <h4>📝 {qd['question']}</h4>
        </div>
        """, unsafe_allow_html=True)

        if not qd.get("answered"):
            # ── Answer widgets per type ──
            if sub_type == "choice" and qd.get("options"):
                opts = qd["options"]
                choice = st.radio(
                    "请选择你的答案：",
                    options=[o["label"] for o in opts],
                    format_func=lambda l: f"{l}. {next((o['text'] for o in opts if o['label'] == l), '')}",
                    key="quiz_choice",
                )
                if st.button("✅ 提交答案", key="submit_choice", type="primary", width="stretch"):
                    st.session_state.quiz_data["user_answer"] = choice
                    st.session_state.quiz_data["answered"] = True
                    st.rerun()

            elif sub_type == "judgment":
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ 正确", key="judge_true", width="stretch", type="primary"):
                        st.session_state.quiz_data["user_answer"] = "正确"
                        st.session_state.quiz_data["answered"] = True
                        st.rerun()
                with c2:
                    if st.button("❌ 错误", key="judge_false", width="stretch"):
                        st.session_state.quiz_data["user_answer"] = "错误"
                        st.session_state.quiz_data["answered"] = True
                        st.rerun()

            elif sub_type == "fill":
                answer = st.text_input("请输入你的答案：", key="fill_answer", placeholder="填入空白处的内容...")
                if st.button("✅ 提交答案", key="submit_fill", type="primary", width="stretch"):
                    if answer.strip():
                        st.session_state.quiz_data["user_answer"] = answer.strip()
                        st.session_state.quiz_data["answered"] = True
                        st.rerun()
                    else:
                        st.warning("请输入答案后再提交")

            elif sub_type == "essay":
                answer = st.text_area("请输入你的回答：", key="essay_answer", height=150,
                                      placeholder="在此输入你的答案...")
                if st.button("📝 提交评分", key="submit_essay", type="primary", width="stretch"):
                    if answer.strip():
                        st.session_state.quiz_data["user_answer"] = answer.strip()
                        st.session_state.quiz_data["answered"] = True
                        st.rerun()
                    else:
                        st.warning("请输入答案后再提交")

        else:
            # ── Evaluate and show result ──
            if qd.get("is_correct") is None:
                with st.spinner("正在评阅..."):
                    try:
                        eval_result = pipeline.quiz_agent.evaluate_answer(
                            question=qd["question"],
                            user_answer=qd["user_answer"],
                            correct_answer=qd["correct_answer"],
                            knowledge_node_ids=qd.get("knowledge_node_ids", []),
                            sub_type=sub_type,
                            user_id="default",
                            options=qd.get("options"),
                        )
                        st.session_state.quiz_data["is_correct"] = eval_result["is_correct"]
                        st.session_state.quiz_data["score"] = eval_result["score"]
                        if sub_type == "essay" and "comment" in eval_result:
                            st.session_state.quiz_data["essay_comment"] = eval_result.get("comment", "")
                        st.rerun()
                    except Exception as e:
                        st.error(f"评阅失败: {e}")

            # Show result
            is_correct = qd.get("is_correct", False)
            score = qd.get("score", 0)

            if sub_type == "essay":
                st.markdown(f"""
                <div class="quiz-card" style="border-left-color: {'#22c55e' if is_correct else '#f59e0b'};">
                    <h4>{'✅' if is_correct else '📝'} 得分: {score:.0%}</h4>
                    <p><strong>你的回答:</strong> {qd['user_answer'][:500]}</p>
                    <p><strong>参考答案:</strong> {qd.get('correct_answer', '')[:500]}</p>
                    {f"<p><strong>评语:</strong> {qd.get('essay_comment', '')}</p>" if qd.get('essay_comment') else ""}
                </div>
                """, unsafe_allow_html=True)
            else:
                icon = "✅" if is_correct else "❌"
                st.markdown(f"""
                <div class="quiz-card" style="border-left-color: {'#22c55e' if is_correct else '#ef4444'};">
                    <h4>{icon} {'回答正确！' if is_correct else '回答错误'}</h4>
                    <p><strong>你的答案:</strong> {qd['user_answer']}</p>
                    <p><strong>正确答案:</strong> {qd.get('correct_answer', '')}</p>
                </div>
                """, unsafe_allow_html=True)

            # Show explanation
            if qd.get("explanation"):
                with st.expander("📖 查看解析", expanded=not is_correct):
                    st.markdown(qd["explanation"])

            # Add to history
            if not any(h.get("question") == qd.get("question") for h in st.session_state.quiz_history):
                st.session_state.quiz_history.append({
                    "question": qd.get("question", "")[:80],
                    "sub_type": sub_type,
                    "is_correct": is_correct,
                    "user_answer": qd.get("user_answer", ""),
                    "correct_answer": qd.get("correct_answer", ""),
                })

            # Next action
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 再来一题", key="quiz_again", width="stretch", type="primary"):
                    st.session_state.quiz_data = None
                    for k in ("quiz_choice", "fill_answer", "essay_answer"):
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()
            with c2:
                if st.button("📋 查看答题记录", key="quiz_history_btn", width="stretch"):
                    st.rerun()

    # ── Quiz History ──
    if st.session_state.quiz_history:
        with st.expander(f"📋 答题记录 ({len(st.session_state.quiz_history)} 题)", expanded=False):
            correct_count = sum(1 for h in st.session_state.quiz_history if h.get("is_correct"))
            total = len(st.session_state.quiz_history)
            st.progress(correct_count / max(total, 1), text=f"正确率: {correct_count}/{total} ({correct_count/max(total,1)*100:.0f}%)")
            for i, h in enumerate(reversed(st.session_state.quiz_history[-20:])):
                icon = "✅" if h.get("is_correct") else "❌"
                st.caption(f"{icon} {i+1}. [{h.get('sub_type', '?')}] {h.get('question', '')[:60]}...")


def render_progress_page():
    st.markdown("## 📊 学习进度")
    pipeline = get_pipeline()

    if not pipeline:
        st.info("系统未初始化")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 掌握度分布")
        try:
            import plotly.graph_objects as go
            summary = pipeline.progress_tracker.get_progress_summary("default")
            mastery_dist = summary.get("mastery_distribution", {})
            total = summary.get("total_knowledge_nodes", 0)
            if total > 0:
                labels = list(mastery_dist.keys())
                values = list(mastery_dist.values())
                colors = {"unknown": "#94a3b8", "exposed": "#fbbf24",
                          "familiar": "#60a5fa", "proficient": "#34d399", "mastered": "#8b5cf6"}
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=[colors.get(l, "#94a3b8") for l in labels]),
                    hole=0.4,
                )])
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("暂无学习数据")
        except Exception as e:
            st.warning(f"获取掌握度失败: {e}")

    with col2:
        st.markdown("### ⏰ 待复习")
        try:
            due = pipeline.progress_tracker.get_due_reviews("default", limit=8)
            if due:
                for d in due:
                    title = d.get("title", d.get("knowledge_node_id", ""))
                    overdue = d.get("overdue_days", 0)
                    mastery = d.get("mastery", "unknown")
                    label = {"unknown": "🆕", "exposed": "📖", "familiar": "📗", "proficient": "📘", "mastered": "📚"}
                    st.markdown(
                        f"<div class='progress-card'><small>{label.get(mastery, '❓')} "
                        f"**{title}** — 逾期 {overdue:.0f}天</small></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("暂无待复习内容")
        except Exception as e:
            st.warning(f"获取复习列表失败: {e}")

    st.markdown("### ❌ 薄弱知识点")
    try:
        weak = pipeline.progress_tracker.get_weak_nodes("default", threshold=1)
        if weak:
            for w in weak[:8]:
                mastery_map = {"unknown": "未学习", "exposed": "已接触", "familiar": "熟悉",
                               "proficient": "熟练", "mastered": "已掌握"}
                ml = mastery_map.get(w["mastery"], w["mastery"])
                st.markdown(
                    f"<div class='progress-card' style='border-left-color:#ef4444;'>"
                    f"⚠️ **{w['title']}** — {ml} · 答错 {w['wrong_count']} 次</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("暂无薄弱知识点 🎉")
    except Exception as e:
        st.warning(f"获取薄弱点失败: {e}")

    st.markdown("### ❌ 错题记录")
    try:
        wrong = pipeline.progress_tracker.get_wrong_answers("default", limit=10)
        if wrong:
            for w in wrong:
                with st.expander(f"❌ {w['question'][:80]}..."):
                    st.markdown(f"**你的回答:** {w['user_answer']}")
                    st.markdown(f"**正确答案:** {w['correct_answer']}")
                    st.markdown(f"**答错次数:** {w['wrong_count']}")
        else:
            st.info("暂无错题记录")
    except Exception as e:
        st.warning(f"获取错题失败: {e}")


def render_knowledge_graph_page():
    st.markdown("## 🔗 知识图谱")
    pipeline = get_pipeline()

    if not pipeline:
        st.info("系统未初始化")
        return

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        _PLOTLY_AVAILABLE = True
    except ImportError:
        _PLOTLY_AVAILABLE = False

    entity_name = st.text_input("搜索实体", placeholder="输入概念名称，如「线性回归」「监督学习」")

    if entity_name:
        with st.spinner("查询中..."):
            info = get_entity_info(entity_name)
            st.markdown(f"<div class='kg-card'>{info}</div>", unsafe_allow_html=True)

    st.markdown("### 🗺️ 知识图谱概览")
    try:
        kg = pipeline.knowledge_graph
        if kg:
            entities = kg.get_all_entities() if hasattr(kg, 'get_all_entities') else []
            total_entities = len(entities)
            types = {}
            for e in entities:
                if isinstance(e, dict):
                    et = e.get("entity_type", e.get("type", "unknown"))
                    types[et] = types.get(et, 0) + 1

            if total_entities > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("总实体数", total_entities)
                with col2:
                    st.metric("总关系数", len(kg.get_all_relations()) if hasattr(kg, 'get_all_relations') else "?")

                if _PLOTLY_AVAILABLE and types:
                    fig = px.bar(
                        x=list(types.keys()),
                        y=list(types.values()),
                        labels={"x": "实体类型", "y": "数量"},
                        color=list(types.values()),
                        color_continuous_scale="Viridis",
                    )
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("知识图谱为空，上传文档后将自动构建")
        else:
            st.info("知识图谱未初始化")
    except Exception as e:
        st.warning(f"获取知识图谱失败: {e}")


# ── Main ──

if not st.session_state.pipeline_ready:
    st.markdown("""
    <div style='text-align:center;padding:3rem 0;'>
        <h1 style='font-size:2.5rem;margin-bottom:0.5rem;'>🧠 AI 学习系统</h1>
        <p style='font-size:1.1rem;color:#6b7280;'>智能问答 · 知识管理 · 自适应学习</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.pipeline_error:
        st.error(st.session_state.pipeline_error)
        st.info("""
        💡 常见原因：
        1. `.env` 文件缺失或配置不正确（检查 API Key 是否有效）
        2. 网络无法连接到 Qdrant/LLM 服务
        3. 外部服务（Qdrant Cloud）连接超时
        
        请在终端中查看详细错误日志。
        """)

    if st.session_state.init_in_progress:
        with st.spinner("正在连接 Qdrant、加载 Embedding 模型..."):
            st.markdown("首次初始化可能需要 10-30 秒，请耐心等待...")
            try:
                pipeline = ensure_pipeline()
                if pipeline:
                    st.session_state.pipeline_ready = True
                    st.session_state.pipeline_error = ""
                    st.session_state.init_in_progress = False
                else:
                    st.session_state.pipeline_error = "初始化返回空，请检查配置"
                    st.session_state.init_in_progress = False
            except Exception as e:
                st.session_state.pipeline_error = f"初始化失败: {e}"
                st.session_state.init_in_progress = False
        if st.session_state.pipeline_ready:
            st.rerun()
    else:
        if st.button("🚀 初始化系统", width="stretch", type="primary"):
            st.session_state.init_in_progress = True
            st.rerun()

elif st.session_state.pipeline_ready:
    page = st.session_state.get("selected_page", "💬 对话")

    if page == "💬 对话":
        render_chat_page()
    elif page == "📄 文档管理":
        render_documents_page()
    elif page == "🧪 出题":
        render_quiz_page()
    elif page == "📊 学习进度":
        render_progress_page()
    elif page == "🔗 知识图谱":
        render_knowledge_graph_page()
