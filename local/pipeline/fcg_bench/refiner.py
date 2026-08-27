"""把结构化的 BenchmarkItem 送给 Claude(Opus)做自然语言润色。

调用方式(需要先安装 anthropic 官方 SDK):
    pip install anthropic

    from fcg_bench.refiner import ClaudeRefiner
    refiner = ClaudeRefiner()              # 读取 ANTHROPIC_API_KEY 环境变量
    refined_items = refiner.refine_batch(items)

设计原则(对应你文档里那句:"Opus 不负责发现事实,只负责表述事实"):

- 只让模型改写 question 和 natural_answer
- 原始 gold_answer 完全保留,作为唯一真相
- 要求模型严格按照 gold 事实来写,不能编造新函数名
- 提供函数源码/文件路径作为上下文,但禁止引入 gold 之外的调用
- 输出强制 JSON,字段 {"question": ..., "answer": ...}
- 验证:生成文本里提到的函数短名必须是 gold 里出现过的子集

这个文件不会在 import 时真的调用 API。不装 anthropic 也能 import。
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable
from .schema import BenchmarkItem


SYSTEM_PROMPT = """你是一个代码知识库的问答数据集撰写助手。

你会收到一条已经由程序从函数调用图中抽取好的结构化事实。
你的唯一任务是:把这条事实改写成一问一答的自然中文,用于评测 GraphRAG 的 local search 能力。

严格规则:
1. 不能引入结构化事实里没有的函数名。所有你提到的函数必须来自 gold 字段。
2. 不能臆造不存在的调用顺序、返回值或异常,保持与 gold 一致。
3. 问题要面向一个不了解代码细节的读者,自然流畅,避免"根据 gold"这种痕迹。
4. 答案要完整覆盖 gold 里的每一个关键实体,但用自然语言串起来,不要只是罗列。
5. 如果 gold 有路径,答案必须按 gold 里的顺序描述。
6. 答案控制在 80-180 字之间。
7. 你的最终输出必须是一个 JSON 对象,且只有这个 JSON,不要任何解释或 Markdown 围栏。
   JSON schema:{"question": "...", "answer": "..."}
"""


def build_user_prompt(item: BenchmarkItem, code_snippet: str = "") -> str:
    parts = [
        f"题型: {item.question_type}",
        f"中心函数: {item.root_function}",
        f"结构化事实 (gold):\n{json.dumps(item.gold_answer, ensure_ascii=False, indent=2)}",
        f"程序生成的草稿问题: {item.question}",
        f"程序生成的草稿答案: {item.natural_answer}",
    ]
    if code_snippet:
        # 限制长度,避免超 token
        snippet = code_snippet[:2000]
        parts.append(f"中心函数源码(仅供风格参考,不要引入新函数名):\n```python\n{snippet}\n```")
    parts.append("请按系统提示输出 JSON。")
    return "\n\n".join(parts)


# ------------ 幻觉校验 ------------

def _extract_identifiers(text: str) -> set[str]:
    """粗略抽文本里像函数名的 token。"""
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text))


def allowed_names(item: BenchmarkItem) -> set[str]:
    """收集 gold 里允许出现的函数短名。"""
    allowed: set[str] = {item.root_function.rsplit(".", 1)[-1]}
    for v in item.gold_answer.values():
        if isinstance(v, str):
            allowed.add(v)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    allowed.add(x)
                elif isinstance(x, list):
                    for y in x:
                        if isinstance(y, str):
                            allowed.add(y)
    return {a.rsplit(".", 1)[-1] for a in allowed if a}


def validate_refined(item: BenchmarkItem, question: str, answer: str) -> tuple[bool, str]:
    """检查润色结果是否出现了 gold 里没有的函数名。允许有少量常见词。"""
    if not question.strip() or not answer.strip():
        return False, "empty question or answer"
    allowed = allowed_names(item)
    # 提取 answer 里所有 snake_case / camelCase 标识符
    tokens = _extract_identifiers(answer) | _extract_identifiers(question)
    # 过滤掉常见词
    STOPWORDS = {
        "函数", "调用", "返回", "异常", "处理", "子图", "路径", "主要", "执行",
        "当", "否则", "成立", "不同", "核心", "辅助", "阶段", "经过", "完成",
        "对象", "结果", "包含", "然后", "接着", "输入", "输出", "生成",
    }
    suspicious = set()
    for t in tokens:
        if t in STOPWORDS:
            continue
        if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", t) is None:
            continue
        # 必须看起来像个函数(下划线或小写开头)才校验
        if not re.search(r"[a-z]", t):
            continue
        if t not in allowed:
            suspicious.add(t)
    if suspicious:
        return False, f"hallucinated identifiers: {sorted(suspicious)[:5]}"
    return True, ""


# ------------ 真正的调用封装 ------------

@dataclass
class RefineConfig:
    model: str = "claude-opus-4-5"         # 你文档里用 Opus 做润色
    max_tokens: int = 600
    temperature: float = 0.2
    max_retries: int = 2


class ClaudeRefiner:
    """薄封装。client 可以由外部注入,便于测试。"""

    def __init__(
        self,
        client: Any | None = None,
        config: RefineConfig | None = None,
        code_fetcher: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config or RefineConfig()
        self.code_fetcher = code_fetcher or (lambda qn: "")
        if client is not None:
            self.client = client
        else:
            try:
                import anthropic
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
            except ImportError:
                self.client = None

    def _call(self, system: str, user: str) -> str:
        if self.client is None:
            raise RuntimeError(
                "未配置 Claude 客户端。请 pip install anthropic 并设置 ANTHROPIC_API_KEY。"
            )
        msg = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # 取第一个 text block
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        text = text.strip()
        # 去掉 markdown 代码围栏
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # 兜底:找第一个 {...} 块
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return None
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(obj, dict):
            return None
        return obj

    def refine(self, item: BenchmarkItem) -> BenchmarkItem:
        code = self.code_fetcher(item.root_function)
        user = build_user_prompt(item, code)
        last_err = ""
        for _ in range(self.config.max_retries + 1):
            try:
                raw = self._call(SYSTEM_PROMPT, user)
            except Exception as e:
                last_err = f"api error: {e}"
                continue
            obj = self._parse_json(raw)
            if obj is None:
                last_err = "json parse failed"
                continue
            q = obj.get("question", "").strip()
            a = obj.get("answer", "").strip()
            ok, reason = validate_refined(item, q, a)
            if not ok:
                last_err = reason
                # 下一轮让模型重试,user 里附上错误提示
                user = build_user_prompt(item, code) + \
                    f"\n\n上次输出被拒绝:{reason}。请严格只使用 gold 中的函数名,重新输出 JSON。"
                continue
            # 成功:原地替换 question 和 natural_answer,gold_answer 不动
            item.question = q
            item.natural_answer = a
            return item
        # 全部失败:保留原样,加一条脚注
        item.natural_answer += f"  [refine_failed: {last_err}]"
        return item

    def refine_batch(self, items: list[BenchmarkItem]) -> list[BenchmarkItem]:
        return [self.refine(it) for it in items]
