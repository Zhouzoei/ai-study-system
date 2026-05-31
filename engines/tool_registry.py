import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]
    handler: Callable
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
        }


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        logger.debug(f"Tool registered: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        tools = self._tools.values()
        if category:
            tools = [t for t in tools if t.category == category]
        return [t.to_dict() for t in tools]

    def call(self, name: str, **kwargs) -> Any:
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        try:
            return tool.handler(**kwargs)
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            raise

    def build_tools_description(self, category: Optional[str] = None) -> str:
        tools = self._tools.values()
        if category:
            tools = [t for t in tools if t.category == category]
        lines = []
        for t in tools:
            params_desc = ", ".join(f"{k}: {v}" for k, v in t.parameters.items())
            lines.append(f"- {t.name}({params_desc}): {t.description}")
        return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """你是一个 AI 学习助手，帮助用户理解知识、巩固记忆。

{learning_context}

## 核心原则
- 严格基于参考资料回答，不编造信息
- 如果资料不足，明确指出哪些部分没有找到
- 引用来源时使用 [来源 X] 格式
- 根据学习进度调整讲解深度"""

QA_PROMPT_TEMPLATE = """{system_prompt}

{conv_text}[参考资料]:
{context_text}

{kg_text}

问题: {question}

请给出详细、准确的回答:"""

QA_SHORT_PROMPT_TEMPLATE = """{system_prompt}

用户说: {message}

请友好地简短回应，保持自然，无需使用参考资料。"""

QUIZ_PROMPT_TEMPLATE = """你是一个专业的学习测试助手。请严格基于以下参考资料出题，不要使用参考资料以外的知识。
{learning_context}
参考资料:
{context_text}

出题主题: {topic}

出题类型: {sub_type}

{sub_type_instructions}

重要：题目和解析必须基于参考资料内容，如果参考资料不足以出题，请说明。"""

QUIZ_TYPE_INSTRUCTIONS = {
    "choice": (
        "格式要求：\n## 题目\n[题目内容]\n\n## 选项\nA. [选项]\nB. [选项]\nC. [选项]\nD. [选项]\n\n"
        "## 正确答案\n[正确选项字母]\n\n## 解析\n[基于参考资料的详细解析，标注来源]"
    ),
    "judgment": (
        "格式要求：\n## 题目\n[判断陈述]\n\n## 答案\n✅ 正确 / ❌ 错误\n\n"
        "## 解析\n[基于参考资料的详细解析，标注来源]"
    ),
    "fill": (
        "格式要求：\n## 题目\n[包含____的题目]\n\n## 答案\n[填空答案]\n\n"
        "## 解析\n[基于参考资料的详细解析，标注来源]"
    ),
    "essay": (
        "格式要求：\n## 题目\n[简答题目]\n\n## 参考答案\n[基于参考资料的参考答案，标注来源]\n\n"
        "## 评分要点\n[列出3-5个得分点]"
    ),
}

SUMMARY_PROMPT_TEMPLATE = """你是一个专业的知识总结助手。请严格基于以下参考资料进行总结，不要添加参考资料中没有的信息。
{learning_context}
{instruction}

参考资料:
{context_text}

总结主题: {topic}

请按以下格式输出总结：

## 核心要点
- [3-5个核心观点，每个观点标注来源]

## 关键术语
- [列出关键术语及其简短定义]

## 知识结构
[用层级列表展示知识点之间的逻辑关系]

## 一句话总结
[用一句话概括核心内容]"""

REVIEW_SCHEDULED_TEMPLATE = """你是一个个性化的学习复习助手。以下是根据间隔重复算法(SM-2)计算出的待复习知识点，请帮助用户进行高效复习。
{learning_context}
待复习知识点:
{review_text}

{f"参考资料:" + chr(10) + context_text if context_text else ""}

请按以下格式生成复习内容：

## 📋 复习概览
[简要说明当前复习状态和重点]

## 🔍 知识点回顾
[对每个待复习知识点，用1-2段话回顾核心内容，帮助用户重新激活记忆]

## 💡 记忆技巧
[针对容易遗忘的知识点，提供记忆口诀或关联记忆方法]

## ✅ 自测问题
[针对待复习知识点，提出2-3个简单的自测问题帮助检验记忆]"""

REVIEW_ASSOCIATED_TEMPLATE = """你是一个个性化学习助手。以下是与「{topic}」关联的知识点，请帮助用户进行关联复习。
{learning_context}
关联知识点:
{assoc_text}

关联关系:
{rel_text}

请按以下格式生成关联复习内容：

## 🔗 知识关联图
[说明这些知识点之间的逻辑关系和依赖]

## 📝 关联复习要点
[对每个关联知识点，说明其与「{topic}」的关系，并回顾核心内容]

## 🧠 知识网络构建
[帮助用户建立这些知识点之间的心智模型]"""

ESSAY_GRADING_TEMPLATE = """请评估以下简答回答的得分。

题目: {question}
参考答案: {correct_answer}
用户回答: {user_answer}

请给出0-1之间的分数（0=完全不正确，1=完全正确）和简要评语。
只输出JSON格式: {{"score": 0.85, "comment": "..."}}"""

ENTITY_EXTRACTION_TEMPLATE = """请从以下文本中提取实体和关系，按JSON格式输出。

文本:
{content}

请按以下格式输出:
{{
  "entities": [
    {{"name": "实体名", "type": "概念/技术/工具/人物/组织", "description": "简短描述"}}
  ],
  "relations": [
    {{"source": "源实体名", "target": "目标实体名", "type": "包含/依赖/属于/相关/基于/实现", "description": "关系描述"}}
  ]
}}

只输出JSON，不要其他内容:"""

DISTILL_PROMPT_TEMPLATE = """从以下文档段落中提取独立的"知识点"。每个知识点应该是一个可以独立学习和评估的原子概念。

输出 JSON 格式，不要其他内容:
{{
  "knowledge_units": [
    {{
      "concept": "概念名称",
      "definition": "精确定义（1-2句话）",
      "bloom_level": "记忆/理解/应用/分析/评价/创造",
      "prerequisites": ["前置概念1", "前置概念2"],
      "keywords": ["关键词1", "关键词2"],
      "difficulty": 0.5,
      "examples": ["示例1"]
    }}
  ]
}}

文档段落:
{batch_text}"""

INTENT_ROUTER_TEMPLATE = """你是一个意图分类器。请分析用户消息，判断其意图类型。

意图类型定义：
- qa: 用户提出问题，希望获得知识性回答
- quiz: 用户希望系统出题测试自己
- summary: 用户希望对某个主题或文档进行总结归纳
- review: 用户希望复习已学知识

用户消息: {message}

请按以下JSON格式输出：
{{"intent": "qa/quiz/summary/review", "confidence": 0.0-1.0, "topic": "提取的主题关键词", "sub_type": "补充类型"}}

sub_type可选值：
- qa: factual/reasoning/exploratory/comparison/procedural
- quiz: choice/judgment/fill/essay
- summary: document/topic/chapter
- review: scheduled/weak_point/associated

只输出JSON，不要其他内容："""

REFLECTION_REWRITE_TEMPLATE = """原始问题: {question}
之前的回答不够充分: {previous_answer}

请改写原始问题，使其更具体、更容易检索到相关信息。只输出改写后的问题，不要其他内容。"""

SAME_CONCEPT_TEMPLATE = """请判断以下两个名称是否指代同一个概念/事物。
只需回答"是"或"否"，不要其他内容。
名称1: {name_a}
名称2: {name_b}"""

CHAT_RESPONSE_TEMPLATE = """用户说: {message}

请友好地简短回应，保持自然，无需使用参考资料。"""


class PromptManager:
    def __init__(self):
        self._templates: Dict[str, str] = {
            "system": SYSTEM_PROMPT_TEMPLATE,
            "qa": QA_PROMPT_TEMPLATE,
            "qa_short": QA_SHORT_PROMPT_TEMPLATE,
            "quiz": QUIZ_PROMPT_TEMPLATE,
            "summary": SUMMARY_PROMPT_TEMPLATE,
            "review_scheduled": REVIEW_SCHEDULED_TEMPLATE,
            "review_associated": REVIEW_ASSOCIATED_TEMPLATE,
            "essay_grading": ESSAY_GRADING_TEMPLATE,
            "entity_extraction": ENTITY_EXTRACTION_TEMPLATE,
            "distill": DISTILL_PROMPT_TEMPLATE,
            "intent_router": INTENT_ROUTER_TEMPLATE,
            "reflection_rewrite": REFLECTION_REWRITE_TEMPLATE,
            "same_concept": SAME_CONCEPT_TEMPLATE,
            "chat_response": CHAT_RESPONSE_TEMPLATE,
        }
        self._type_instructions = dict(QUIZ_TYPE_INSTRUCTIONS)

    def render(self, name: str, **kwargs) -> str:
        template = self._templates.get(name)
        if not template:
            raise ValueError(f"Prompt template not found: {name}")
        if name == "quiz":
            sub_type = kwargs.get("sub_type", "choice")
            kwargs["sub_type_instructions"] = self._type_instructions.get(sub_type, self._type_instructions["choice"])
        if name == "summary":
            sub_type = kwargs.get("sub_type", "topic")
            type_instructions = {
                "document": "请对以下文档内容进行全面总结。",
                "chapter": "请对以下章节内容进行重点总结。",
                "topic": "请对以下关于指定主题的内容进行总结归纳。",
            }
            kwargs["instruction"] = type_instructions.get(sub_type, type_instructions["topic"])
        return template.format(**kwargs)

    def register_template(self, name: str, template: str):
        self._templates[name] = template
