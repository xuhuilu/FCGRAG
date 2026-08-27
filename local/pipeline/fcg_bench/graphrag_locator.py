"""
GraphRAG 的 root function 定位模块。

对应 LocAgent 的 SearchEntity 阶段 —— 从 question 文本出发,找到正确的 root function。

两阶段流程:
  1. 候选检索 (retrieve_candidates): 基于关键词匹配,不需要 LLM,<10ms
  2. LLM 精排 (rank_candidates):    把候选喂给 LLM,选一个最匹配的

输出: ranked list of qualnames。第一个就是最可能的 root。

和 LocAgent 的差异:
  - LocAgent 用 BM25 索引 + 多轮 agent loop
  - 我们用正则关键词 + 单轮 LLM 精排
  原因:你的题目文本里几乎总直接提到函数 short name,不需要复杂的语义检索
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any

from .schema import CallGraph
from .graphrag_rag import LLMClient, parse_llm_json


# ============================================================
# 关键词抽取
# ============================================================

# 匹配"看起来像函数/类名"的 token
# 1. 驼峰: ChatAnthropic, ToolMessage
# 2. 下划线: handle_request, _parse_input
# 3. 限定名: BaseTool.run, foo.bar
_IDENT_RE = re.compile(
    r"""
    (?:
        [A-Z][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*   # BaseTool.run
      | [a-z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*  # foo.bar  (少见)
      | [A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]+           # 驼峰 2+ 段 ChatAnthropic
      | [a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+         # 含下划线 handle_request
    )
    """,
    re.VERBOSE,
)

# 噪声词汇过滤:这些词经常出现在 question 文本里但不是函数名
_STOPWORDS = {
    "is_not", "is_none", "not_none",           # 常见条件片段
    "value_error", "type_error", "key_error",  # 异常类的 snake 形式
    "init_py", "__init__", "__main__",
    "true_path", "false_path",                 # 题目里 gold 字段的描述词
    "error_path", "gold_path",
    "root_function", "question_type",
    "call_graph",
}


def extract_keywords_from_question(question: str) -> list[str]:
    """从题目文本里抽所有"看起来像函数名"的 token,去重保序。"""
    found = _IDENT_RE.findall(question)
    seen = set()
    result = []
    for tok in found:
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        result.append(tok)
    return result


# ============================================================
# 候选检索
# ============================================================

@dataclass
class Candidate:
    qualname: str
    short_name: str
    file_path: str
    docstring_1line: str
    match_score: float          # 越大越相关
    match_reason: str           # 调试用:为什么这个候选被选上

    def to_prompt_line(self) -> str:
        doc = self.docstring_1line[:80]
        doc_part = f" — {doc}" if doc else ""
        return f"  - {self.qualname}{doc_part}"


class RootLocator:
    """
    两阶段 root 定位器。

    用法:
        locator = RootLocator(cg, llm)
        root_qn, candidates = locator.locate(question)
    """

    def __init__(self, cg: CallGraph, llm: LLMClient | None = None):
        self.cg = cg
        self.llm = llm

        # 建立 short_name → list[qualname] 索引(预计算)
        self._short_to_qns: dict[str, list[str]] = {}
        self._short_to_qns_lower: dict[str, list[str]] = {}
        for qn, node in cg.nodes.items():
            short = node.short_name
            self._short_to_qns.setdefault(short, []).append(qn)
            self._short_to_qns_lower.setdefault(short.lower(), []).append(qn)

    # -------- 阶段 1:纯规则候选检索 --------
    def retrieve_candidates(
        self, question: str, max_candidates: int = 20,
    ) -> list[Candidate]:
        """
        从题目里抽关键词,在图里找匹配的函数节点,返回打分后的候选列表。

        评分规则(累加):
          +3.0  短名精确匹配某个关键词(大小写敏感)
          +2.0  短名大小写不敏感匹配
          +1.0  短名是某个关键词的子串(或反之)
          +1.5  关键词是"Class.method"形式且 qualname 精确包含这个子串
          +0.5  docstring 第一行包含关键词
        """
        keywords = extract_keywords_from_question(question)
        if not keywords:
            return []

        # qualname -> (score, reasons)
        scores: dict[str, tuple[float, list[str]]] = {}

        def bump(qn: str, delta: float, reason: str):
            if qn not in self.cg.nodes:
                return
            old = scores.get(qn, (0.0, []))
            scores[qn] = (old[0] + delta, old[1] + [reason])

        for kw in keywords:
            # 1. short_name 精确匹配(大小写敏感)
            for qn in self._short_to_qns.get(kw, []):
                bump(qn, 3.0, f"exact:{kw}")

            # 2. 大小写不敏感匹配
            for qn in self._short_to_qns_lower.get(kw.lower(), []):
                if kw not in self._short_to_qns:  # 避免重复加分
                    bump(qn, 2.0, f"nocase:{kw}")

            # 3. Class.method 形式:在 qualname 里找匹配
            if "." in kw:
                suffix = kw  # e.g. "BaseTool.run"
                for qn in self.cg.nodes:
                    if qn.endswith("." + suffix) or qn.endswith(suffix):
                        bump(qn, 1.5, f"qualified:{kw}")

            # 4. 子串匹配(宽松,放后面防止噪声)
            kw_low = kw.lower()
            for short_low, qns in self._short_to_qns_lower.items():
                if short_low == kw_low:
                    continue
                if len(kw_low) >= 5 and (kw_low in short_low or short_low in kw_low):
                    for qn in qns:
                        bump(qn, 1.0, f"substr:{kw}")

        if not scores:
            return []

        # 按分数排序,取前 max_candidates 个
        ranked = sorted(scores.items(), key=lambda kv: -kv[1][0])[:max_candidates]

        candidates: list[Candidate] = []
        for qn, (score, reasons) in ranked:
            node = self.cg.nodes[qn]
            doc = (node.docstring or "").split("\n")[0].strip()
            candidates.append(Candidate(
                qualname=qn,
                short_name=node.short_name,
                file_path=node.file_path,
                docstring_1line=doc,
                match_score=score,
                match_reason=",".join(reasons[:3]),
            ))
        return candidates

    # -------- 阶段 2:LLM 精排 --------
    def rank_candidates(
        self,
        question: str,
        candidates: list[Candidate],
        top_k: int = 5,
    ) -> list[str]:
        """
        把候选列表喂给 LLM,让它输出 ranked list。
        返回 list[qualname],按相关性降序。
        如果 LLM 不可用或出错,退化为按 match_score 排序。
        """
        if not candidates:
            return []
        if self.llm is None or len(candidates) == 1:
            return [c.qualname for c in candidates[:top_k]]

        system = """你是一个代码定位助手。根据用户问题,从候选函数列表里选出最相关的函数。

你必须以 JSON 格式输出一个 ranked list,字段为 "ranked_qualnames"。
只输出 JSON,不要任何解释或 markdown 标记。
"""
        candidates_text = "\n".join(c.to_prompt_line() for c in candidates[:20])
        user = f"""# 用户问题
{question}

# 候选函数(按初步匹配分数排序)
{candidates_text}

# 任务
从上面的候选里,选出**最可能是这道题要分析的 root function** 的那些函数,按相关性降序排列。
注意:
1. 题目里直接提到的函数名(比如 "从 X 开始..."、"X 中的..."、"X 的返回值..." 里的 X)是最强信号
2. 如果题目问某个分支/异常/主链,选那个"被讨论的函数",不是分支里的子函数
3. 最多返回 top {top_k} 个 qualname

输出 JSON:
{{"ranked_qualnames": ["pkg.mod.Func1", "pkg.mod.Func2", ...]}}
"""
        try:
            raw = self.llm.complete(system, user, temperature=0.0)
            parsed = parse_llm_json(raw)
            ranked = parsed.get("ranked_qualnames", [])
            # 过滤掉不在候选里的 qualname
            valid_qns = {c.qualname for c in candidates}
            ranked = [qn for qn in ranked if qn in valid_qns]
            if ranked:
                return ranked[:top_k]
        except Exception:
            pass

        # 回退:按规则分数排序
        return [c.qualname for c in candidates[:top_k]]

    # -------- 一站式入口 --------
    def locate(
        self, question: str, top_k: int = 5,
    ) -> tuple[str | None, list[Candidate], list[str]]:
        """
        端到端定位。
        返回:
          (best_root_qn, candidates, ranked_qns)
          best_root_qn 可能为 None(完全定位失败)
        """
        candidates = self.retrieve_candidates(question)
        if not candidates:
            return None, [], []

        ranked_qns = self.rank_candidates(question, candidates, top_k=top_k)
        best = ranked_qns[0] if ranked_qns else None
        return best, candidates, ranked_qns
