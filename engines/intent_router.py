import re
import json
from typing import List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class IntentDef:
    label: str
    description: str
    keywords: List[str] = field(default_factory=list)

    def match(self, text: str) -> float:
        text_lower = text.lower()
        matches = sum(1 for kw in self.keywords if kw.lower() in text_lower)
        return min(matches * 0.35, 1.0)


class IntentType:
    QA = IntentDef("qa", "用户提出问题，希望获得知识性回答",
                   keywords=["什么", "怎么", "为什么", "如何", "区别", "对比",
                             "原理", "定义", "概念", "是啥", "含义", "例子",
                             "what", "how", "why", "difference", "principle"])
    QUIZ = IntentDef("quiz", "用户希望系统出题测试自己",
                     keywords=["出题", "考考", "考我", "测验", "测试我",
                               "来道题", "来题", "选择题", "判断题", "填空题",
                               "quiz", "test me", "practice"])
    SUMMARY = IntentDef("summary", "用户希望对某个主题或文档进行总结归纳",
                        keywords=["总结", "概括", "摘要", "归纳", "提炼",
                                  "总结一下", "概括一下", "内容摘要", "归纳要点",
                                  "summarize", "summary", "tldr"])
    REVIEW = IntentDef("review", "用户希望复习已学知识",
                       keywords=["复习", "回顾", "温习", "复习一下",
                                 "待复习", "间隔复习", "检查记忆",
                                 "review", "revise", "recall"])
    CHAT = IntentDef("chat", "用户进行日常聊天、打招呼、寒暄或表达情感，不需要知识点回答",
                     keywords=["你好", "嗨", "hello", "hi", "早上好", "晚上好",
                               "谢谢", "感谢", "再见", "拜拜", "bye",
                               "哈哈", "不错", "很好", "好的", "ok"])
    TUTOR = IntentDef("tutor", "用户希望系统引导学习、制定学习计划或推荐学习内容",
                      keywords=["学习计划", "学习路线", "怎么学", "学习建议",
                                "帮我规划", "推荐内容", "下一步学什么",
                                "study plan", "learning path"])
    EXPLAIN = IntentDef("explain", "用户要求系统用通俗易懂的方式解释复杂概念",
                        keywords=["通俗解释", "简单解释", "比喻", "举例说明",
                                  "白话", "讲清楚", "形象地", "打个比方",
                                  "explain simply", "analogy", "ELI5"])
    COMPARE = IntentDef("compare", "用户要求系统对比两个或多个概念、技术或工具",
                        keywords=["对比", "比较", "vs", "versus", "区别",
                                  "差异", "优缺点", "哪个好", "选哪个",
                                  "compare", "difference between"])

    _ALL = [QA, QUIZ, SUMMARY, REVIEW, CHAT, TUTOR, EXPLAIN, COMPARE]


@dataclass
class IntentResult:
    intent: IntentDef = field(default_factory=lambda: IntentType.QA)
    confidence: float = 0.5
    topic: str = ""
    sub_type: str = ""
    raw_response: str = ""


INTENT_PROMPT = """你是一个高精度意图分类器。请分析用户消息，判断其最可能的意图类型。

## 意图类型定义
- qa: 用户提出问题，希望获得知识性回答。例如："什么是机器学习？"、"Python怎么实现排序？"
- quiz: 用户希望系统出题测试自己。例如："出个机器学习的选择题"、"考考我"
- summary: 用户希望对某个主题或文档进行总结归纳。例如："总结一下这篇文章"、"概括监督学习"
- review: 用户希望复习已学知识。例如："帮我复习一下线性回归"、"今天要复习什么"
- chat: 用户进行日常聊天、打招呼、寒暄或表达情感。例如："你好"、"谢谢"、"再见"
- tutor: 用户希望系统引导学习、制定学习计划。例如："帮我规划学习路线"、"下一步学什么"
- explain: 用户要求通俗解释复杂概念。例如："用比喻解释神经网络"、"白话讲什么是注意力机制"
- compare: 用户要求对比两个或多个概念。例如："对比CNN和RNN"、"PyTorch和TensorFlow哪个好"

## 示例
用户: 什么是过拟合？
输出: {"intent": "qa", "confidence": 0.95, "topic": "过拟合", "sub_type": "factual"}

用户: 来道关于线性回归的题
输出: {"intent": "quiz", "confidence": 0.9, "topic": "线性回归", "sub_type": "choice"}

用户: 总结一下Transformer
输出: {"intent": "summary", "confidence": 0.9, "topic": "Transformer", "sub_type": "topic"}

用户: 帮我复习一下昨天的内容
输出: {"intent": "review", "confidence": 0.85, "topic": "复习", "sub_type": "scheduled"}

用户: 你好呀
输出: {"intent": "chat", "confidence": 0.95, "topic": "", "sub_type": "greeting"}

用户: 用比喻解释什么是反向传播
输出: {"intent": "explain", "confidence": 0.9, "topic": "反向传播", "sub_type": "analogy"}

用户: 对比一下随机森林和决策树
输出: {"intent": "compare", "confidence": 0.95, "topic": "随机森林 vs 决策树", "sub_type": "comparison"}

用户: 帮我制定一个深度学习的学习计划
输出: {"intent": "tutor", "confidence": 0.9, "topic": "深度学习", "sub_type": "planning"}

## sub_type 说明
- qa: factual(事实)/reasoning(推理)/exploratory(探索)/procedural(步骤)
- quiz: choice(选择)/judgment(判断)/fill(填空)/essay(简答)
- summary: document(文档)/topic(主题)/chapter(章节)
- review: scheduled(定期)/weak_point(薄弱点)/associated(关联)
- chat: greeting(问候)/feedback(反馈)/farewell(告别)/general(一般)
- tutor: planning(规划)/recommend(推荐)/guidance(引导)
- explain: analogy(比喻)/simplify(简化)/example(举例)
- compare: comparison(对比)/advantage(优缺点)/selection(选择)

用户消息: {message}

请只输出JSON格式，不要其他内容：
{{"intent": "intent类型", "confidence": 0.0-1.0, "topic": "主题关键词", "sub_type": "补充类型"}}"""


def _infer_sub_type(intent: IntentDef, text: str) -> str:
    if intent == IntentType.QA:
        if re.search(r'(?:什么|是什么|定义|含义|概念)', text):
            return "factual"
        if re.search(r'(?:为什么|原因|原理|机制)', text):
            return "reasoning"
        if re.search(r'(?:比较|对比|vs|区别)', text):
            return "comparison"
        if re.search(r'(?:步骤|流程|教程|如何)', text):
            return "procedural"
        return "factual"
    if intent == IntentType.QUIZ:
        if re.search(r'选择题', text):
            return "choice"
        if re.search(r'判断题', text):
            return "judgment"
        if re.search(r'填空题', text):
            return "fill"
        return "choice"
    if intent == IntentType.SUMMARY:
        if re.search(r'文档|文件|这篇', text):
            return "document"
        if re.search(r'章节|第\d+章', text):
            return "chapter"
        return "topic"
    if intent == IntentType.REVIEW:
        return "scheduled"
    if intent == IntentType.CHAT:
        if re.search(r'你好|嗨|hello|hi|早上好|晚上好|^hi', text.lower()):
            return "greeting"
        if re.search(r'谢谢|感谢|多谢|thank', text.lower()):
            return "feedback"
        if re.search(r'再见|拜拜|bye|下次', text.lower()):
            return "farewell"
        return "general"
    if intent == IntentType.TUTOR:
        if re.search(r'计划|规划|路线', text):
            return "planning"
        if re.search(r'推荐|建议|学什么', text):
            return "recommend"
        return "guidance"
    if intent == IntentType.EXPLAIN:
        if re.search(r'比喻|类比|打个比方|形象', text):
            return "analogy"
        if re.search(r'白话|简单|通俗|简化', text):
            return "simplify"
        return "example"
    if intent == IntentType.COMPARE:
        return "comparison"
    return ""


def _extract_topic(text: str) -> str:
    stop_words = {
        "的", "了", "是", "在", "有", "和", "与", "或", "不", "也",
        "什么", "怎么", "如何", "为什么", "哪里", "哪个", "谁",
        "出", "出个", "帮我", "给我", "总结", "概括", "复习",
        "一下", "一道", "一个", "考考", "考我",
        "你好", "谢谢", "再见", "hello", "hi",
    }
    tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{2,}', text)
    meaningful = [t for t in tokens if t not in stop_words]
    return meaningful[0] if meaningful else ""


class IntentRouter:
    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func

    def route(self, message: str) -> IntentResult:
        if self.llm_func:
            result = self._llm_route(message)
            if result and result.confidence >= 0.5:
                return result

        return self._rule_route(message)

    def _llm_route(self, message: str) -> Optional[IntentResult]:
        prompt = INTENT_PROMPT.replace("{message}", message[:800])
        try:
            response = self.llm_func(prompt)
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(clean)

            intent_str = parsed.get("intent", "qa").lower()
            intent = IntentType.QA
            for it in IntentType._ALL:
                if it.label == intent_str:
                    intent = it
                    break

            confidence = min(max(float(parsed.get("confidence", 0.5)), 0.0), 1.0)
            topic = parsed.get("topic", "")
            sub_type = parsed.get("sub_type", "")

            return IntentResult(
                intent=intent, confidence=confidence,
                topic=topic, sub_type=sub_type, raw_response=response,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return None

    def _rule_route(self, message: str) -> IntentResult:
        msg = message.strip()
        if not msg:
            return IntentResult(intent=IntentType.QA, confidence=0.5)

        best_intent = IntentType.QA
        best_score = 0.0
        for intent in IntentType._ALL:
            if intent == IntentType.QA:
                continue
            score = intent.match(msg)
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_score < 0.3:
            return IntentResult(
                intent=IntentType.QA,
                confidence=0.7,
                topic=_extract_topic(msg),
                sub_type=_infer_sub_type(IntentType.QA, msg),
            )

        return IntentResult(
            intent=best_intent,
            confidence=min(best_score + 0.3, 1.0),
            topic=_extract_topic(msg),
            sub_type=_infer_sub_type(best_intent, msg),
        )
