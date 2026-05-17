import os
import sys
import json
import time
import uuid
import datetime
import re
import threading
from typing import Optional

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.pipeline import EnhancedRAGPipeline
from engines.adaptive_retriever import QueryClassifier
from utils.llm_service import LLMService
from utils.embedding_service import EmbeddingService

_PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    pass

_global_lock = threading.Lock()
_pipeline_instance: Optional[EnhancedRAGPipeline] = None
_embed_service: Optional[EmbeddingService] = None

APP_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="*neutral_50", block_background_fill="white",
    block_border_color="*neutral_200", block_radius="14px",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_200",
    input_background_fill="white", input_border_color="*neutral_200",
    input_radius="10px", panel_background_fill="white",
    border_color_primary="*primary_200",
)

CSS = """
.container { max-width: 1200px; margin: auto; }
footer { display: none !important; }
.header-wrap { text-align: center; padding: 1.6rem 0 0.4rem 0; margin-bottom: 0.5rem; border-bottom: 1px solid #e5e7eb; }
.header-wrap h1 {
    font-size: 1.8rem !important; font-weight: 700 !important; margin: 0 0 0.3rem 0 !important;
    background: linear-gradient(135deg, #4338ca, #6366f1, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.header-wrap .sub { font-size: 0.88rem; color: #6b7280; letter-spacing: 0.01em; }
.nav-bar { display: flex; gap: 4px; margin-bottom: 12px; flex-wrap: wrap; }
.nav-btn { flex: 1; min-width: 100px; text-align: center; font-weight: 500 !important; border-radius: 10px 10px 0 0 !important; }
.gr-box { border-radius: 14px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06); border: 1px solid #e5e7eb; }
.gr-accordion { border-radius: 12px !important; border: 1px solid #e5e7eb !important; margin-bottom: 8px !important; }
.gr-accordion > .label-wrap { font-weight: 500 !important; }
.quick-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
.quick-chips button {
    font-size: 0.78rem !important; padding: 4px 14px !important;
    border-radius: 20px !important; border: 1px solid #d1d5db !important;
    background: #f9fafb !important; color: #374151 !important;
    transition: all 0.15s !important; font-weight: 450 !important; box-shadow: none !important;
}
.quick-chips button:hover { background: #eef2ff !important; border-color: #a5b4fc !important; color: #3730a3 !important; }
.gr-chatbot { border-radius: 14px !important; overflow: hidden !important; }
.gr-chatbot .message.bot { background: #f8fafc !important; border: 1px solid #e2e8f0 !important; border-radius: 0 14px 14px 14px !important; padding: 12px 16px !important; }
.gr-chatbot .message.user { background: #eef2ff !important; border: 1px solid #c7d2fe !important; border-radius: 14px 0 14px 14px !important; padding: 12px 16px !important; }
.kg-card { background: #f8fafc; border-radius: 12px; padding: 16px 20px; margin-bottom: 10px; border-left: 4px solid #6366f1; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.kg-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.kg-card h4 { margin: 0 0 8px 0; color: #1e293b; font-size: 1rem; }
.kg-card p { margin: 3px 0; font-size: 0.85rem; color: #475569; line-height: 1.5; }
.kg-card em { color: #6366f1; font-style: normal; font-weight: 500; }
.progress-card { background: #f0fdf4; border-radius: 12px; padding: 16px 20px; margin-bottom: 8px; border-left: 4px solid #22c55e; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.progress-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.progress-card h4 { margin: 0 0 8px 0; color: #14532d; }
.progress-card p { margin: 3px 0; font-size: 0.85rem; color: #166534; }
.progress-bar-wrap { background: #d1fae5; border-radius: 6px; height: 8px; margin-top: 8px; overflow: hidden; }
.progress-bar-fill { background: linear-gradient(90deg, #6366f1, #22c55e); height: 100%; border-radius: 6px; transition: width 0.4s ease; }
.stat-line { font-family: "SF Mono", "Cascadia Code", "JetBrains Mono", ui-monospace, monospace; font-size: 0.8rem; line-height: 1.65; color: #334155; }
.doc-card { background: white; border-radius: 10px; padding: 12px 16px; margin-bottom: 6px; border: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.doc-card:hover { border-color: #a5b4fc; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
.fade-in { animation: fadeIn 0.3s ease; }
"""


def get_pipeline():
    global _pipeline_instance
    with _global_lock:
        if _pipeline_instance is not None:
            return _pipeline_instance
    return None


def ensure_pipeline():
    global _pipeline_instance, _embed_service
    with _global_lock:
        if _pipeline_instance is not None:
            return _pipeline_instance
        _embed_service = EmbeddingService()
        llm_service = LLMService()
        _pipeline_instance = EnhancedRAGPipeline(
            embed_func=_embed_service.embed,
            llm_func=llm_service.invoke,
            llm_service=llm_service,
        )
        return _pipeline_instance


def create_session():
    pipeline = get_pipeline()
    if pipeline:
        session = pipeline.create_session()
        return session["session_id"]
    return f"sess_{uuid.uuid4().hex[:12]}"


def get_stats():
    pipeline = get_pipeline()
    if not pipeline:
        return {"storage": {}, "conversation": {}, "learning_progress": {}}
    try:
        return pipeline.get_stats()
    except Exception:
        return {"error": "获取统计失败"}


def _build_stats_lines():
    stats = get_stats()
    storage = stats.get("storage", {})
    conv = stats.get("conversation", {})
    lc = storage.get("level_counts", {})
    kg = stats.get("knowledge_graph", {})
    return "\n".join([
        f"节点: {storage.get('total_nodes', 0)} (L1={lc.get(1,0)}, L2={lc.get(2,0)}, L3={lc.get(3,0)})",
        f"文档数: {storage.get('doc_count', 0)}",
        f"对话数: {conv.get('total_sessions', 0)}",
        f"消息数: {conv.get('total_messages', 0)}",
        f"知识图谱: {kg.get('total_entities', 0)} 实体, {kg.get('total_relations', 0)} 关系",
    ])


# ── Format helpers ──

def _fmt_entity_html(result: dict) -> str:
    if not result or "error" in result:
        return f"<div class='kg-card'><p style='color:#ef4444;'>{result.get('error', '无结果')}</p></div>"
    entity = result.get("entity", result)
    if not entity:
        return "<div class='kg-card'><p>未找到实体</p></div>"
    name = entity.get("name", result.get("entity_name", "未知实体"))
    etype = entity.get("entity_type", entity.get("type", ""))
    desc = entity.get("description", "")
    props = entity.get("properties", {})
    relations = result.get("relations", [])
    neighbors = result.get("neighbors", [])
    _id_to_name = {entity.get("entity_id", ""): name}
    for n in neighbors:
        if isinstance(n, dict):
            _id_to_name[n.get("entity_id", "")] = n.get("name", "?")
    html = f"""<div class='kg-card'><h4>🔍 {name}</h4>"""
    if etype:
        html += f"<p><strong>类型:</strong> {etype}</p>"
    if desc:
        html += f"<p>{desc[:300]}</p>"
    if props:
        html += "<p><strong>属性:</strong> " + ", ".join(f"{k}: {v}" for k, v in list(props.items())[:8]) + "</p>"
    html += "</div>"
    if relations:
        html += "<div class='kg-card'><h4>🔗 关系列表</h4>"
        for r in relations[:15]:
            if isinstance(r, dict):
                rel_type = r.get("relation_type", r.get("type", r.get("relation", "?")))
                src_id = r.get("source_entity_id", "")
                tgt_id = r.get("target_entity_id", "")
                desc_r = r.get("description", "")
                src_name = _id_to_name.get(src_id, src_id[:12] if src_id else "?")
                tgt_name = _id_to_name.get(tgt_id, tgt_id[:12] if tgt_id else "?")
                html += f"<p>→ <strong>{src_name}</strong> — <em>{rel_type}</em> → <strong>{tgt_name}</strong></p>"
                if desc_r:
                    html += f"<p style='font-size:0.8rem;color:#888;'>{desc_r[:100]}</p>"
            else:
                html += f"<p>→ {r}</p>"
        html += "</div>"
    if neighbors:
        html += "<div class='kg-card'><h4>👥 相邻实体</h4><p>"
        for n in neighbors[:20]:
            if isinstance(n, dict):
                nname = n.get("name", "?")
                ntype = n.get("entity_type", "")
                label = f"{nname}({ntype})" if ntype else nname
                html += f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:2px 10px;border-radius:12px;background:#eef2ff;color:#4338ca;font-size:0.82rem;'>{label}</span>"
            else:
                html += str(n) + ", "
        html += "</p></div>"
    return html


def _fmt_multi_hop_html(result: list) -> str:
    if not result:
        return "<div class='kg-card'><p>未找到路径</p></div>"
    if isinstance(result, dict) and "error" in result:
        return f"<div class='kg-card'><p style='color:#ef4444;'>{result['error']}</p></div>"
    html = ""
    for i, path_data in enumerate(result):
        if isinstance(path_data, dict) and "path" in path_data:
            path = path_data["path"]
            rels = path_data.get("relations", [])
            hops = path_data.get("hops", 0)
            html += f"<div class='kg-card'><h4>🛤️ 路径 {i+1} ({hops} 跳)</h4>"
            for j, node in enumerate(path):
                nname = node.get("name", "?") if isinstance(node, dict) else str(node)
                html += f"<p><strong>{nname}</strong></p>"
                if j < len(rels):
                    rel = rels[j]
                    rtype = rel.get("type", "?") if isinstance(rel, dict) else str(rel)
                    html += f"<p style='color:#6366f1;padding-left:12px;'>↓ <em>{rtype}</em></p>"
            html += "</div>"
        elif isinstance(path_data, dict):
            src = path_data.get("source", path_data.get("from", "?"))
            rel = path_data.get("relation", path_data.get("type", "?"))
            tgt = path_data.get("target", path_data.get("to", "?"))
            html += f"<div class='kg-card'><p><strong>{i+1}.</strong> {src} — <em>{rel}</em> → {tgt}</p></div>"
        else:
            html += f"<div class='kg-card'><p><strong>{i+1}.</strong> {path_data}</p></div>"
    if not html:
        html = f"<div class='kg-card'><pre>{json.dumps(result, ensure_ascii=False, indent=2)}</pre></div>"
    return html


def _fmt_progress_html(progress: dict) -> str:
    if not progress or "error" in progress:
        return f"<div class='progress-card'><p style='color:#ef4444;'>{progress.get('error', '暂无数据')}</p></div>"
    total = progress.get("total_knowledge_nodes", progress.get("total_nodes", 0))
    if total == 0:
        return "<div class='progress-card' style='text-align:center;'><h4>📭 暂无学习记录</h4><p style='color:#6b7280;'>请先上传学习文档</p></div>"
    mastery_dist = progress.get("mastery_distribution", {})
    mastered = mastery_dist.get("mastered", 0) + mastery_dist.get("proficient", 0)
    learning = mastery_dist.get("learning", 0) + mastery_dist.get("familiar", 0)
    unknown = mastery_dist.get("unknown", 0)
    due_count = progress.get("due_for_review", 0)
    overall_pct = progress.get("progress_pct", round(mastered / max(total, 1) * 100))
    total_exposures = progress.get("total_exposures", 0)
    avg_ease = progress.get("avg_ease_factor", 2.5)
    html = f"""<div class='progress-card'><h4>📊 总体进度</h4><p>已掌握: <strong>{mastered}</strong> / 学习中: <strong>{learning}</strong> / 未学习: <strong>{unknown}</strong> / 总计: <strong>{total}</strong></p><div class='progress-bar-wrap'><div class='progress-bar-fill' style='width:{overall_pct}%'></div></div><p style='font-size:0.8rem;color:#888;'>{overall_pct}% 已掌握 · 平均复习间隔: {avg_ease:.1f}天 · 待复习: {due_count}个</p></div>"""
    if mastery_dist:
        html += "<div class='progress-card'><h4>📋 掌握度分布</h4>"
        labels = {"mastered": "已掌握", "proficient": "熟练", "learning": "学习中", "familiar": "熟悉", "unknown": "未学习"}
        for level, count in mastery_dist.items():
            label = labels.get(level, level)
            pct = round(count / max(total, 1) * 100)
            html += f"<p><strong>{label}</strong>: {count} ({pct}%)</p>"
        html += "</div>"
    return html


def _fmt_due_html(due_list: list) -> str:
    if not due_list:
        return "<div class='progress-card'><p>🎉 暂无待复习知识点</p></div>"
    if isinstance(due_list, dict) and "error" in due_list:
        return f"<div class='progress-card'><p style='color:#ef4444;'>{due_list['error']}</p></div>"
    html = ""
    for i, item in enumerate(due_list[:15]):
        if isinstance(item, dict):
            title = item.get("title", "") or item.get("knowledge_node_id", f"知识点 {i+1}")
            mastery = item.get("mastery", "")
            exposures = item.get("exposure_count", 0)
            overdue = item.get("overdue_days", 0)
            next_review_ts = item.get("next_review_at", 0)
            html += f"<div class='progress-card'><p><strong>{i+1}. {title}</strong></p>"
            if mastery:
                html += f"<p style='font-size:0.8rem;color:#888;'>掌握度: {mastery} · 复习次数: {exposures}</p>"
            if next_review_ts and next_review_ts > 0:
                try:
                    dt = datetime.datetime.fromtimestamp(next_review_ts)
                    html += f"<p style='font-size:0.8rem;color:#888;'>应复习: {dt.strftime('%m-%d %H:%M')}</p>"
                except Exception:
                    pass
            if overdue > 0:
                html += f"<p style='font-size:0.8rem;color:#ef4444;'>已逾期 {overdue} 天</p>"
            html += "</div>"
        else:
            html += f"<div class='progress-card'><p><strong>{i+1}.</strong> {item}</p></div>"
    return html


def _fmt_dashboard_html(dashboard: dict) -> str:
    if not dashboard or "error" in dashboard:
        return f"<div class='progress-card'><p style='color:#ef4444;'>{dashboard.get('error', '暂无数据')}</p></div>"
    progress = dashboard.get("progress", {})
    coverage = dashboard.get("knowledge_coverage", {})
    weaknesses = dashboard.get("weaknesses", {})
    plan_summary = dashboard.get("plan_summary", {})
    html = "<div class='progress-card'><h4>📊 学习概览</h4>"
    html += f"<p>知识覆盖率: <strong>{coverage.get('coverage_pct', 0)}%</strong> ({coverage.get('covered_nodes', 0)}/{coverage.get('total_knowledge_nodes', 0)})</p>"
    html += f"<p>待复习: <strong>{progress.get('due_reviews_count', 0)}</strong> · 进行中计划: <strong>{plan_summary.get('active_plans', 0)}</strong></p>"
    html += "</div>"
    if weaknesses.get("forgotten_areas"):
        html += "<div class='progress-card'><h4>⚠️ 已遗忘的知识点</h4>"
        for area in weaknesses["forgotten_areas"][:3]:
            html += f"<p>• {area.get('title', '?')} (逾期 {area.get('overdue_days', 0)} 天)</p>"
        html += "</div>"
    if weaknesses.get("weak_areas"):
        html += "<div class='progress-card'><h4>💪 薄弱知识点</h4>"
        for area in weaknesses["weak_areas"][:3]:
            html += f"<p>• {area.get('title', '?')} ({area.get('reason', '')})</p>"
        html += "</div>"
    return html


def _fmt_recommendations_html(recs: list) -> str:
    if not recs:
        return "<div class='progress-card'><p>🎉 暂无学习建议</p></div>"
    html = ""
    for r in recs[:8]:
        priority = r.get("priority", "medium")
        color = "#ef4444" if priority == "high" else "#f59e0b" if priority == "medium" else "#6b7280"
        html += f"<div class='progress-card' style='border-left-color:{color};'><p><strong>{r.get('title', '?')}</strong></p><p style='font-size:0.8rem;color:#888;'>{r.get('reason', '')}</p><p style='font-size:0.8rem;color:{color};'>建议: {r.get('action', '')}</p></div>"
    return html


# ── Dual-Loop helpers ──

def _detect_intent(message: str) -> Optional[str]:
    msg = message.strip()
    quiz_patterns = [
        r"^(出|出个|出一[道个]|给我|帮我).*(题|考题|题目|选择题|判断题|填空题|问题)",
        r"(出题|出个题|来道题|来题|考我|测验|测试)",
        r"(quiz|test me|practice question)",
    ]
    for p in quiz_patterns:
        if re.search(p, msg, re.IGNORECASE):
            return "quiz"

    summary_patterns = [
        r"^(总结|概括|摘要|归纳|提炼).*",
        r".*(总结一下|概括一下|帮我总结|做个总结|内容摘要)",
        r"(summarize|summary|TLDR|tl;dr)",
    ]
    for p in summary_patterns:
        if re.search(p, msg, re.IGNORECASE):
            return "summarize"

    return None


def _fmt_analysis(analysis) -> str:
    type_labels = {
        "factual": "事实型",
        "reasoning": "推理型",
        "exploratory": "探索型",
        "comparison": "比较型",
        "procedural": "步骤型",
    }
    qtype = analysis.query_type.value if hasattr(analysis.query_type, "value") else str(analysis.query_type)
    label = type_labels.get(qtype, qtype)
    kw = analysis.keywords[:5] if analysis.keywords else []
    entities = analysis.entities[:3] if analysis.entities else []
    kw_str = "、".join(kw) if kw else "—"
    ent_str = "、".join(entities) if entities else "—"
    strat = analysis.suggested_strategy
    topk = analysis.suggested_top_k
    return (
        f"🤔 **问题分析**\n"
        f"- 类型: `{label}` (置信度: {analysis.confidence:.0%})\n"
        f"- 意图: {analysis.intent}\n"
        f"- 关键词: {kw_str}\n"
        f"- 检索: `{strat}` (top-{topk})\n"
    )


def _handle_quiz(message: str) -> str:
    pipeline = get_pipeline()
    if not pipeline or not pipeline.llm_service:
        return "⚠️ 系统未就绪，无法出题"
    prompt = f"""你是一个智能出题助手。请根据用户的要求生成一道题目。

用户要求: {message}

请按以下格式输出:
## 题目
[题目内容]

## 选项
A. [选项]
B. [选项]
C. [选项]
D. [选项]

## 正确答案
[正确选项]

## 解析
[简要解析]

如果用户没指定知识点，出一道机器学习的基础题。"""
    try:
        return pipeline.llm_service.invoke(prompt)
    except Exception as e:
        return f"⚠️ 出题失败: {e}"


def _handle_summarize(message: str) -> str:
    pipeline = get_pipeline()
    if not pipeline or not pipeline.llm_service:
        return "⚠️ 系统未就绪，无法总结"
    prompt = f"""用户要求: {message}

请生成一个结构化的摘要，包含:
- 核心观点（3-5点）
- 关键术语
- 一句话总结

如果没有指定具体内容，请回复"请指定要总结的内容"。"""
    try:
        return pipeline.llm_service.invoke(prompt)
    except Exception as e:
        return f"⚠️ 总结失败: {e}"


# ── Core handlers ──

def chat_respond(message: str, history: list, session_state: str, mode: str = "hybrid"):
    pipeline = get_pipeline()
    if not pipeline:
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "系统未初始化，请稍候重试"})
        yield history, ""
        return

    session_id = session_state or f"sess_{uuid.uuid4().hex[:12]}"
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield history, ""

    # === DUAL-LOOP PHASE 1: INTENT DETECTION ===
    intent = _detect_intent(message)
    if intent == "quiz":
        result = _handle_quiz(message)
        history[-1]["content"] = f"🎯 **智能出题**\n\n{result}"
        yield history, ""
        return
    elif intent == "summarize":
        result = _handle_summarize(message)
        history[-1]["content"] = f"📋 **智能总结**\n\n{result}"
        yield history, ""
        return

    # === DUAL-LOOP PHASE 2: ANALYSIS ===
    classifier = QueryClassifier()
    analysis = classifier.classify(message)
    analysis_text = _fmt_analysis(analysis)
    history[-1]["content"] = analysis_text + "\n📚 **正在检索知识库...**"
    yield history, ""

    # === DUAL-LOOP PHASE 3: SOLVE (RAG) ===
    resolved_mode = mode
    if mode == "hybrid" and analysis.suggested_strategy == "hybrid" and analysis.suggested_top_k > 5:
        resolved_mode = "deep"

    pipeline_result = pipeline.query(
        message, session_id=session_id, mode=resolved_mode,
    )

    if not pipeline_result["contexts"]:
        history[-1]["content"] = analysis_text + "⚠️ 暂无相关文档内容。请先上传文档后再提问。\n\n"
        yield history, ""
        return

    mode_tag = pipeline_result.get("strategy", "")
    citations = ""
    for i, chain in enumerate(pipeline_result["context_chains"][:3]):
        l1 = chain.get("l1_title", "") or ""
        l2 = chain.get("l2_title", "") or ""
        l3 = chain.get("l3_title", "") or ""
        citations += f"\n> 参考 {i+1}: {l1} / {l2} / {l3[:40]}"

    prefix = f"{analysis_text}`⚙️ {mode_tag}`\n\n{citations}\n\n"
    history[-1]["content"] = prefix
    yield history, ""
    partial = ""
    for chunk in pipeline.generate_answer_stream(message, pipeline_result):
        partial += chunk
        history[-1]["content"] = prefix + partial
        yield history, ""

    sources = pipeline_result.get("context_sources", [])
    if sources:
        source_block = "\n\n---\n\n📖 **来源原文**\n\n"
        for s in sources[:3]:
            idx = s.get("index", "?")
            title = s.get("level_title", "")
            excerpt = s.get("excerpt", "")
            source_block += f"**来源 {idx}** — {title}\n\n> {excerpt}\n\n"
        history[-1]["content"] = prefix + partial + source_block
        yield history, ""


def ingest_document(text: str, title: str, doc_id: str, progress=gr.Progress()):
    pipeline = ensure_pipeline()
    progress(0.2, "分层分块中...")
    result = pipeline.ingest(text, doc_id=doc_id, title=title)
    progress(1.0, "完成")
    l3 = result["level_counts"].get(3, 0)
    l2 = result["level_counts"].get(2, 0)
    l1 = result["level_counts"].get(1, 0)
    return (
        f"导入成功！\n"
        f"  文档: {title}\n"
        f"  节点: {result['total_nodes']} (L1={l1}, L2={l2}, L3={l3})\n"
        f"  知识图谱: {result['kg_entities']} 实体, {result['kg_relations']} 关系\n"
        f"  耗时: {result['chunk_time_ms']:.0f}ms 分块 + {result['store_time_ms']:.0f}ms 存储"
    )


def upload_and_ingest(file, title: str, progress=gr.Progress()):
    if file is None:
        return "请先上传文件"
    content = ""
    if file.name.endswith((".txt", ".md")):
        with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    else:
        try:
            from pypdf import PdfReader
            reader = PdfReader(file.name)
            for page in reader.pages:
                content += page.extract_text() + "\n"
        except ImportError:
            return "未安装pypdf，仅支持.txt/.md文件"
        except Exception as e:
            return f"PDF读取失败: {e}"
    if not content.strip():
        return "文件内容为空"
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    title = title or os.path.splitext(os.path.basename(file.name))[0]
    return ingest_document(content, title, doc_id, progress)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI 学习系统") as app:
        gr.HTML("""<div class="header-wrap fade-in"><h1>AI 学习系统</h1><div class="sub">上传学习资料 → 自动构建知识体系 → 智能问答与复习</div></div>""")

        # ── Navigation ──
        with gr.Row(elem_classes="nav-bar"):
            btn_qa = gr.Button("💬 智能问答", variant="primary", size="lg", elem_classes="nav-btn")
            btn_docs = gr.Button("📄 文档管理", variant="secondary", size="lg", elem_classes="nav-btn")
            btn_learning = gr.Button("📈 学习中心", variant="secondary", size="lg", elem_classes="nav-btn")

        session_state = gr.State("")

        # ════════════════════════════
        # QA Section
        # ════════════════════════════
        with gr.Column(visible=True) as qa_section:
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("##### 💡 快捷提问")
                    with gr.Row(elem_classes=["quick-chips"]):
                        chip_1 = gr.Button("什么是机器学习？", size="sm", scale=0)
                        chip_2 = gr.Button("深度学习 vs 传统方法", size="sm", scale=0)
                        chip_3 = gr.Button("神经网络原理", size="sm", scale=0)
                        chip_4 = gr.Button("注意力机制解释", size="sm", scale=0)
                    chatbot = gr.Chatbot(
                        label="对话", height=520,
                        placeholder="👋 欢迎使用 AI 学习系统！上传文档后即可开始智能问答。",
                    )
                    chat_input = gr.Textbox(label="输入问题", placeholder="输入你的问题...")
                    mode_selector = gr.Radio(
                        choices=[("⚡ 快速 naive", "naive"), ("🎯 均衡 hybrid", "hybrid"), ("🔬 深度 deep", "deep")],
                        value="hybrid",
                        label="检索策略",
                        info="naive=纯向量, hybrid=向量+BM25, deep=混合+重排序+改写",
                    )
                    chat_input.submit(
                        fn=chat_respond,
                        inputs=[chat_input, chatbot, session_state, mode_selector],
                        outputs=[chatbot, chat_input],
                    )
                with gr.Column(scale=1):
                    with gr.Accordion("📋 会话信息", open=True):
                        session_id_display = gr.Textbox(label="会话 ID", value="", interactive=False)
                        new_session_btn = gr.Button("🔄 新建会话", variant="secondary", size="sm")
                    with gr.Accordion("📄 文档导入", open=True):
                        doc_upload = gr.File(label="上传文档", file_types=[".txt", ".md", ".pdf"])
                        doc_title = gr.Textbox(label="文档标题（可选）", placeholder="留空则使用文件名")
                        ingest_btn = gr.Button("🚀 导入文档", variant="primary")
                        doc_status = gr.Textbox(label="导入状态", value="尚未导入文档", lines=3, interactive=False, elem_classes=["stat-line"])
                    with gr.Accordion("📊 系统状态", open=False):
                        stats_box = gr.Textbox(label="统计信息", value="加载中...", lines=6, interactive=False, elem_classes=["stat-line"])
                        stats_refresh_btn = gr.Button("🔄 刷新统计", size="sm")

        # ════════════════════════════
        # Docs Section
        # ════════════════════════════
        with gr.Column(visible=False) as docs_section:
            gr.Markdown("### 📄 文档列表")
            doc_list_btn = gr.Button("🔄 刷新文档列表", variant="secondary", size="sm")
            doc_list_display = gr.HTML(label="文档列表", value="<div class='kg-card'><p>点击刷新按钮查看已导入的文档</p></div>")
            with gr.Row():
                delete_doc_id = gr.Textbox(label="文档ID（删除用）", placeholder="输入文档ID", scale=3)
                delete_doc_btn = gr.Button("🗑️ 删除", variant="secondary", scale=1)
            delete_status = gr.Textbox(label="操作结果", value="", lines=1, interactive=False)
            gr.Markdown("---")
            gr.Markdown("### 🔗 知识图谱")
            with gr.Accordion("🔍 实体查询", open=False):
                with gr.Row():
                    kg_entity_input = gr.Textbox(label="实体名称", placeholder="例如: 神经网络、ResNet", scale=3)
                    query_entity_btn = gr.Button("🔍 查询", variant="primary", scale=1)
                entity_result_html = gr.HTML(label="查询结果")
                with gr.Accordion("📋 原始数据", open=False):
                    entity_result_json = gr.JSON(label="JSON")
            with gr.Accordion("🛤️ 多跳查询", open=False):
                with gr.Row():
                    source_entity = gr.Textbox(label="源实体", placeholder="例如: ResNet")
                    target_entity = gr.Textbox(label="目标实体", placeholder="例如: 梯度消失")
                multi_hop_btn = gr.Button("🔗 查询路径", variant="secondary")
                multi_hop_result_html = gr.HTML(label="路径结果")
                with gr.Accordion("📋 原始数据", open=False):
                    multi_hop_result_json = gr.JSON(label="JSON")

        # ════════════════════════════
        # Learning Section
        # ════════════════════════════
        with gr.Column(visible=False) as learning_section:
            gr.Markdown("### 📊 学习仪表盘")
            dashboard_btn = gr.Button("🔄 刷新仪表盘", variant="primary", size="sm")
            dashboard_display = gr.HTML(label="仪表盘", value="<div class='progress-card'><p>点击刷新查看学习概览</p></div>")
            gr.Markdown("### 📈 学习进度")
            with gr.Row():
                progress_btn = gr.Button("🔄 刷新学习进度", variant="secondary", size="sm")
                due_btn = gr.Button("📝 待复习列表", variant="secondary", size="sm")
            progress_display = gr.HTML(label="学习进度")
            progress_json = gr.JSON(label="进度数据", visible=False)
            due_display = gr.HTML(label="待复习知识点")
            due_json = gr.JSON(label="复习数据", visible=False)
            gr.Markdown("### 💡 学习建议")
            rec_btn = gr.Button("🔄 获取学习建议", variant="secondary", size="sm")
            rec_display = gr.HTML(label="学习建议")
            if _PLOTLY_AVAILABLE:
                gr.Markdown("### 📉 掌握度分布图")
                plot_btn = gr.Button("🔄 生成图表", variant="secondary", size="sm")
                plot_display = gr.Plot(label="掌握度分布")

        # ════════════════════════════
        # Event Handlers
        # ════════════════════════════

        def _init_on_load():
            sid = create_session()
            return sid, sid, _build_stats_lines()

        app.load(fn=_init_on_load, outputs=[session_state, session_id_display, stats_box])

        # ── Navigation ──
        def _nav_to(section: str):
            return (
                gr.update(visible=(section == "qa")),
                gr.update(visible=(section == "docs")),
                gr.update(visible=(section == "learning")),
                gr.update(variant="primary" if section == "qa" else "secondary"),
                gr.update(variant="primary" if section == "docs" else "secondary"),
                gr.update(variant="primary" if section == "learning" else "secondary"),
            )

        btn_qa.click(
            fn=lambda: _nav_to("qa"),
            outputs=[qa_section, docs_section, learning_section,
                     btn_qa, btn_docs, btn_learning],
        )
        btn_docs.click(
            fn=lambda: _nav_to("docs"),
            outputs=[qa_section, docs_section, learning_section,
                     btn_qa, btn_docs, btn_learning],
        )
        btn_learning.click(
            fn=lambda: _nav_to("learning"),
            outputs=[qa_section, docs_section, learning_section,
                     btn_qa, btn_docs, btn_learning],
        )

        # ── QA handlers ──
        new_session_btn.click(
            fn=lambda: (create_session(), create_session()),
            outputs=[session_state, session_id_display],
        )
        ingest_btn.click(
            fn=upload_and_ingest,
            inputs=[doc_upload, doc_title],
            outputs=[doc_status, stats_box],
        )
        stats_refresh_btn.click(fn=_build_stats_lines, outputs=[stats_box])
        chip_1.click(fn=lambda: "什么是机器学习？", outputs=[chat_input])
        chip_2.click(fn=lambda: "深度学习与传统方法相比有什么优势和劣势？", outputs=[chat_input])
        chip_3.click(fn=lambda: "请解释神经网络的基本原理", outputs=[chat_input])
        chip_4.click(fn=lambda: "请解释注意力机制的原理和应用", outputs=[chat_input])

        # ── Docs handlers ──
        def handle_list_docs():
            pipeline = get_pipeline()
            if not pipeline:
                return "<div class='kg-card'><p>系统未初始化</p></div>"
            try:
                docs = pipeline.list_documents(limit=50)
                if not docs:
                    return "<div class='kg-card'><p>暂无文档，请在智能问答页面上传</p></div>"
                html = ""
                for d in docs:
                    title = d.get("title", d.get("doc_id", "?"))
                    doc_id = d.get("doc_id", "?")
                    tags = d.get("tags", [])
                    tag_html = " ".join(
                        f"<span style='display:inline-block;padding:0 8px;border-radius:8px;background:#eef2ff;color:#4338ca;font-size:0.75rem;'>{t}</span>"
                        for t in tags[:3]
                    )
                    html += f"<div class='doc-card'><p><strong>{title}</strong> <span style='font-size:0.75rem;color:#888;'>({doc_id})</span> {tag_html}</p></div>"
                return html
            except Exception as e:
                return f"<div class='kg-card'><p style='color:#ef4444;'>获取文档列表失败: {e}</p></div>"

        doc_list_btn.click(fn=handle_list_docs, outputs=[doc_list_display])

        def handle_delete_doc(doc_id):
            pipeline = get_pipeline()
            if not pipeline or not doc_id:
                return "请提供文档ID"
            try:
                pipeline.delete_document(doc_id)
                return f"已删除文档: {doc_id}"
            except Exception as e:
                return f"删除失败: {e}"

        delete_doc_btn.click(fn=handle_delete_doc, inputs=[delete_doc_id], outputs=[delete_status])

        def handle_query_entity(name):
            pipeline = get_pipeline()
            if not pipeline:
                return _fmt_entity_html({"error": "系统未初始化"}), {"error": "系统未初始化"}
            if not name or not name.strip():
                return _fmt_entity_html({"error": "请输入实体名称"}), {"error": "请输入实体名称"}
            try:
                result = pipeline.query_knowledge_graph(name.strip())
                if not result or not result.get("entity"):
                    kg_stats = pipeline.knowledge_graph.get_graph_stats()
                    total = kg_stats.get("total_entities", 0)
                    if total == 0:
                        err = {"error": "知识图谱为空，请先上传文档"}
                        return _fmt_entity_html(err), err
                    suggestions = pipeline.search_kg_entities(name.strip(), limit=8)
                    if suggestions:
                        names = "、".join(f"「{s['name']}」" for s in suggestions)
                        err = {"error": f"未找到「{name}」，你是不是想找：{names}"}
                    else:
                        err = {"error": f"未找到与「{name}」相关的实体"}
                    return _fmt_entity_html(err), err
                return _fmt_entity_html(result), result
            except Exception as e:
                err = {"error": f"查询失败: {e}"}
                return _fmt_entity_html(err), err

        query_entity_btn.click(
            fn=handle_query_entity,
            inputs=[kg_entity_input],
            outputs=[entity_result_html, entity_result_json],
        )

        def handle_multi_hop(src, tgt):
            pipeline = get_pipeline()
            if not pipeline:
                return _fmt_multi_hop_html([{"error": "系统未初始化"}]), {"error": "系统未初始化"}
            if not src or not tgt:
                return _fmt_multi_hop_html([{"error": "请填写源实体和目标实体"}]), {"error": "请填写参数"}
            try:
                result = pipeline.multi_hop_query(src, tgt)
                return _fmt_multi_hop_html(result), result
            except Exception as e:
                err = {"error": f"多跳查询失败: {e}"}
                return _fmt_multi_hop_html(err), err

        multi_hop_btn.click(
            fn=handle_multi_hop,
            inputs=[source_entity, target_entity],
            outputs=[multi_hop_result_html, multi_hop_result_json],
        )

        # ── Learning handlers ──
        def handle_dashboard():
            pipeline = get_pipeline()
            if not pipeline:
                return "<div class='progress-card'><p>系统未初始化</p></div>"
            try:
                dashboard = pipeline.get_learning_dashboard()
                return _fmt_dashboard_html(dashboard)
            except Exception as e:
                return f"<div class='progress-card'><p style='color:#ef4444;'>获取仪表盘失败: {e}</p></div>"

        dashboard_btn.click(fn=handle_dashboard, outputs=[dashboard_display])

        def handle_progress():
            pipeline = get_pipeline()
            if not pipeline:
                return _fmt_progress_html({"error": "系统未初始化"}), {"error": "系统未初始化"}
            try:
                progress = pipeline.get_learning_progress()
                return _fmt_progress_html(progress), progress
            except Exception as e:
                return _fmt_progress_html({"error": f"获取失败: {e}"}), {"error": str(e)}

        progress_btn.click(
            fn=handle_progress,
            outputs=[progress_display, progress_json],
        )

        def handle_due():
            pipeline = get_pipeline()
            if not pipeline:
                return _fmt_due_html([]), []
            try:
                result = pipeline.get_due_reviews()
                return _fmt_due_html(result), result
            except Exception as e:
                return _fmt_due_html([]), []

        due_btn.click(
            fn=handle_due,
            outputs=[due_display, due_json],
        )

        def handle_recommendations():
            pipeline = get_pipeline()
            if not pipeline:
                return "<div class='progress-card'><p>系统未初始化</p></div>"
            try:
                recs = pipeline.get_study_recommendations()
                return _fmt_recommendations_html(recs)
            except Exception as e:
                return f"<div class='progress-card'><p style='color:#ef4444;'>获取建议失败: {e}</p></div>"

        rec_btn.click(fn=handle_recommendations, outputs=[rec_display])

        if _PLOTLY_AVAILABLE:
            def handle_plot():
                pipeline = get_pipeline()
                if not pipeline:
                    return None
                try:
                    progress = pipeline.get_learning_progress()
                    mastery_dist = progress.get("mastery_distribution", {})
                    if not mastery_dist:
                        return None
                    labels_map = {"mastered": "已掌握", "proficient": "熟练", "learning": "学习中", "familiar": "熟悉", "unknown": "未学习"}
                    labels = [labels_map.get(k, k) for k in mastery_dist.keys()]
                    values = list(mastery_dist.values())
                    colors = ["#22c55e", "#86efac", "#facc15", "#f97316", "#d1d5db"]
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors[:len(labels)]), textinfo="label+percent")])
                    fig.update_layout(title="掌握度分布", height=350, margin=dict(l=20, r=20, t=40, b=20))
                    return fig
                except Exception:
                    return None

            plot_btn.click(fn=handle_plot, outputs=[plot_display])

    return app


if __name__ == "__main__":
    import webbrowser
    ensure_pipeline()
    import atexit
    atexit.register(lambda: get_pipeline() and get_pipeline().close())
    print("[App] 系统初始化完成，启动 Web 界面...")
    app = build_app()
    webbrowser.open("http://127.0.0.1:7861")
    app.launch(server_name="127.0.0.1", server_port=7861, share=False, show_error=True, css=CSS, theme=APP_THEME)
