"""
GraphRAG 的题型感知 prompt 模板。

核心设计:每类题一个 prompt,只给 LLM 必要的上下文切片。
这对应方案 2("题型感知的上下文裁剪"),比统一大 prompt 节省 token,
也比 agent 循环稳定。

每个 prompt 返回一对 (system_msg, user_msg),由主流程发给 LLM。
LLM 的输出是严格的 JSON,字段和 evaluator 期待的 gold 字段一致。
"""
from __future__ import annotations
import json
from typing import Any

from .graphrag_tools import LocalContext


# ============================================================
# 通用系统消息
# ============================================================

SYSTEM_COMMON = """你是一个代码结构分析助手。你的任务是根据给定的 Python 函数源码和它的局部调用子图,回答关于该函数局部结构的问题。

你必须以 JSON 格式输出,严格遵守:
1. 严格按照用户要求的 JSON schema 输出
2. 只输出 JSON,不要有任何解释、思考过程或 markdown 代码块标记
3. 所有函数名使用 short name(不含模块路径前缀)
4. 过滤掉明显的噪声调用:isinstance, issubclass, cast, super, len, print, 内置类型构造函数
5. 每个回答都必须包含 "explanation" 字段:用 2-4 句自然语言解释你的答案,说明"为什么"而不只是列举函数名
"""


# ============================================================
# 辅助:把 LocalContext 的常用字段格式化成字符串块
# ============================================================

def _format_source(ctx: LocalContext, max_lines: int = 80) -> str:
    """截断源码到前 N 行。"""
    lines = ctx.root_source.splitlines()
    if len(lines) <= max_lines:
        return ctx.root_source
    head = "\n".join(lines[:max_lines])
    return head + f"\n# ... ({len(lines) - max_lines} more lines truncated)"


def _format_callees(ctx: LocalContext, max_n: int = 15) -> str:
    """把 callee 列表格式化成一行一个。"""
    lines = []
    for c in ctx.callees[:max_n]:
        doc = c["docstring_1line"]
        doc_part = f"  — {doc}" if doc else ""
        lines.append(f"  - {c['short_name']}  [{c['role_hint']}]{doc_part}")
    if len(ctx.callees) > max_n:
        lines.append(f"  - ... ({len(ctx.callees) - max_n} more)")
    return "\n".join(lines)


# ============================================================
# 题型 1:execution_flow
# ============================================================

def prompt_execution_flow(ctx: LocalContext, question: str) -> tuple[str, str]:
    # Deprecated prompt variant: emits a single ``path_nodes`` representative path.
    # The current GPT-5.5 SU8 pipeline instead elicits the branch-aware gold_v2.1
    # schema via run_three_methods_clean_v1_en.py::build_prediction_prompt and
    # build_structure_schema; this function is retained for reference only.
    user = f"""# 题目
{question}

# Root 函数
{ctx.root_short}  (qualname: {ctx.root_qualname})

# 预计算的主链(按源码顺序,已经跳过了分支内的调用)
{' → '.join(ctx.main_path)}

# 树形子图(带控制流标注)
{ctx.subgraph_tree}

# Root 源码(前 80 行)
```python
{_format_source(ctx)}
```

# 要求
基于上述信息,识别这个函数当前可恢复的一条代表性执行路径。选择 3~6 个**按源码顺序**的关键业务函数
(跳过内置函数、日志、纯辅助函数),构造一条抽取路径。不要声称它是唯一主流程。

输出 JSON:
{{
  "path_type": "representative_path",
  "path_condition": "",
  "path_nodes": ["{ctx.root_short}", "func2", "func3", ...],
  "coverage_note": "This is one extracted path, not the unique full execution flow.",
  "explanation": "2-4 句自然语言解释:这条抽取路径的整体目的、每步的作用、前后步骤间的因果关系"
}}
"""
    return SYSTEM_COMMON, user


# ============================================================
# 题型 2:result_origin
# ============================================================

# def prompt_result_origin(ctx: LocalContext, question: str) -> tuple[str, str]:
#     user = f"""# 题目
# {question}

# # Root 函数
# {ctx.root_short}

# # AST 识别出的 return 信息
# - return 表达式: `{ctx.return_expr}`
# - return 里的最外层函数调用: `{ctx.return_source_func}`

# # Root 源码(前 80 行)
# ```python
# {_format_source(ctx)}
# ```

# # 任务说明
# 识别返回值的来源函数,以及该返回值变量被哪些后续函数消费。

# # 关键定义
# - **source_entity**: 第一次产生 target 变量的函数。如果源码里有 `result = foo(...)`, 那 source 就是 `foo`。
# - **consumer_entities**: 在 target 变量被赋值之后, 把它**作为参数**传给的函数。注意:
#   * `bar(result)` 里 bar 是 consumer
#   * `result.method()` 里 method **不是** consumer (result 是 receiver, 不是 arg)
#   * 如果 target 是直接 return 的(没有名字), consumer 一定是空列表

# # Few-shot 示例

# ## 示例 1
# 源码:
# ```python
# def process_request(req):
#     result = parse_input(req)
#     log_metrics(result, level="info")
#     return build_response(result)
# ```
# 题目: "在 process_request 中, 最终返回的 result 是在哪一步生成的?"
# 正确答案:
# {{
#   "target_object": "result",
#   "source_entity": "parse_input",
#   "consumer_entities": ["log_metrics", "build_response"]
# }}
# 解释: result 由 parse_input 产生; 之后被 log_metrics(作为第一个参数) 和 build_response(作为参数) 消费。

# ## 示例 2
# 源码:
# ```python
# def get_config(name):
#     return CONFIG_MAP.get(name)
# ```
# 题目: "get_config 的返回结果来自哪里?"
# 正确答案:
# {{
#   "target_object": "返回值",
#   "source_entity": "get",
#   "consumer_entities": []
# }}
# 解释: 直接 return, 没有给变量命名, 所以 consumer 是空。

# ## 示例 3
# 源码:
# ```python
# def make_chain(model):
#     chain = RunnableLambda(lambda x: x)
#     chain = chain.with_config(tags=["chain"])
#     chain = chain | OutputParser()
#     return chain
# ```
# 题目: "在 make_chain 中, chain 是从哪里产生的, 之后被谁使用?"
# 正确答案:
# {{
#   "target_object": "chain",
#   "source_entity": "RunnableLambda",
#   "consumer_entities": ["with_config"]
# }}
# 解释: chain 由 RunnableLambda 产生; with_config 是被调用在 chain 上的方法但 chain.with_config 形式里 with_config 接收 chain 作为隐式 self, 所以也算; 注意 OutputParser() 是构造另一个对象, 没有把 chain 作为参数, 不算 consumer。

# # 现在请回答上面的题目, 严格输出 JSON

# 输出 JSON:
# {{
#   "target_object": "变量名或'返回值'",
#   "source_entity": "最早生成它的函数名",
#   "consumer_entities": ["后续消费它的函数", ...]
# }}
# """
#     return SYSTEM_COMMON, user


def prompt_result_origin(ctx: LocalContext, question: str) -> tuple[str, str]:
    user = f"""# 题目
{question}

# Root 函数
{ctx.root_short}

# AST 识别出的 return 信息
- return 表达式: `{ctx.return_expr}`
- return 里的最外层函数调用: `{ctx.return_source_func}`

# Root 源码(前 80 行)
```python
{_format_source(ctx)}
```

# 任务

识别“最终返回对象”是由谁最早产生的，以及这个对象在后续是否被别的调用显式消费。

术语定义
target_object:
如果最后返回的是一个已命名变量，如 return result，则填该变量名，例如 "result"
如果最后返回的是一个未命名表达式，如 return foo(x)，则填 "返回值"
source_entity:
第一次产生 target_object 的调用名
可以是普通函数名、方法名、构造器名
但必须是当前 root 源码里实际出现过的名字
例子:
result = parse_input(req) -> source_entity = "parse_input"
return CONFIG_MAP.get(name) -> source_entity = "get"
obj = Builder() -> source_entity = "Builder"
consumer_entities:
在 target_object 被产生之后，把 target_object 作为显式参数传入的函数/方法名列表
只有 target_object 出现在调用参数列表中，才算 consumer
例子:
bar(result) -> "bar" 是 consumer
obj.use(result, mode="x") -> "use" 是 consumer
result.method() -> "method" 不是 consumer
x = wrapper(result) -> "wrapper" 是 consumer
如果 target_object 是直接 return 的未命名表达式，则 consumer_entities 必须是空列表 []
强约束
只能输出当前 root 源码中实际出现过的名字
不要输出 few-shot 示例里的名字，除非它们真的出现在当前源码中
不要根据框架常识补全当前源码中没有出现的函数名、方法名或类名
target.method(...) 这种 receiver 调用里，method 不是 consumer
consumer_entities 按源码中的出现顺序输出，并去重
若无法确定，优先选择当前源码中最直接、最字面的那个调用名，不要臆测更高层封装函数
Never output a function, method, or class name that does not literally appear in the provided root source code.
解释约束(非常重要)
- explanation 只能围绕你最终输出的 target_object / source_entity / consumer_entities 来写
- 不要在 explanation 中引入新的变量名; 特别是不要把返回对象改写成源码里另一个变量名
- 如果 consumer_entities 为空，只能解释为“没有被其他函数显式作为参数消费”或“直接返回”，不要扩展成“内部还被继续使用”
- 不要把 receiver method 调用(`obj.method()`)说成 consumer
- 不要把源码中出现但未出现在最终 JSON 里的函数说成“真正来源”或“真正消费者”
Few-shot 示例
示例 1

源码:

def process_request(req):
    result = parse_input(req)
    log_metrics(result, level="info")
    return build_response(result)

题目: "在 process_request 中, 最终返回的 result 是在哪一步生成的?"
正确答案:
{{
"target_object": "result",
"source_entity": "parse_input",
"consumer_entities": ["log_metrics", "build_response"]
}}

示例 2

源码:

def get_config(name):
    return CONFIG_MAP.get(name)

题目: "get_config 的返回结果来自哪里?"
正确答案:
{{
"target_object": "返回值",
"source_entity": "get",
"consumer_entities": []
}}

示例 3

源码:

def finalize(data):
    temp = normalize(data)
    record(temp)
    cache_value = store(temp)
    return temp

题目: "在 finalize 中, 最终返回的 temp 是从哪里产生的, 之后被谁消费?"
正确答案:
{{
"target_object": "temp",
"source_entity": "normalize",
"consumer_entities": ["record", "store"]
}}

现在请回答上面的题目

严格输出 JSON，不要附加解释，不要输出 Markdown，不要输出示例中的名字，除非它们真的出现在当前源码中。

输出 JSON:
{{
"target_object": "变量名或'返回值'",
"source_entity": "当前源码中实际出现的最早来源调用名",
"consumer_entities": ["当前源码中实际出现、且显式接收 target_object 作为参数的调用名", ...],
"explanation": "2-4 句自然语言解释:该返回值如何产生、数据流向、被谁消费及为什么"
}}
"""
    return SYSTEM_COMMON, user


# ============================================================
# 题型 3:conditional_path
# ============================================================

def prompt_conditional_path(
    ctx: LocalContext,
    question: str,
    branch_details: list[dict[str, Any]],
) -> tuple[str, str]:
    branches_text = json.dumps(branch_details, ensure_ascii=False, indent=2)

    user = f"""# 题目
{question}

# Root 函数
{ctx.root_short}

# 该函数所有 if/else 分支的原始信息
{branches_text}

# Root 源码(前 80 行)
```python
{_format_source(ctx)}
```

# 要求
题目会问某个条件相关的分支路径。你需要:

1. 从题目文本里识别出它关心的条件表达式
2. 在上面的 branches 列表里找到条件最匹配、边界最清楚的那一项
3. 清洗该项的 path_calls:
   - 去除 isinstance/issubclass/cast/len/print/内置类型等噪声
   - 去重(保留首次出现顺序)
   - 最多保留 5 个函数
4. 输出 JSON

输出 JSON:
{{
  "condition": "条件表达式原文",
  "branch_kind": "if_true|if_false",
  "branch_span": {{"start_line": 0, "end_line": 0}},
  "path_nodes": ["{ctx.root_short}", "...该分支中的关键调用"],
  "excluded_branches": ["未覆盖的其他分支标签"],
  "explanation": "2-4 句自然语言解释:当前抽取分支在什么条件下会进入、该分支做了什么,并明确这不代表所有其他分支都等价"
}}
"""
    return SYSTEM_COMMON, user


# ============================================================
# 题型 4:error_handling_path
# ============================================================

def prompt_error_handling(
    ctx: LocalContext,
    question: str,
    try_details: list[dict[str, Any]],
) -> tuple[str, str]:
    tries_text = json.dumps(try_details, ensure_ascii=False, indent=2)

    user = f"""# 题目
{question}

# Root 函数
{ctx.root_short}

# 该函数所有 try/except 块的原始信息
{tries_text}

# Root 源码(前 80 行)
```python
{_format_source(ctx)}
```

# 要求
题目会问"当某个函数抛出异常时,经过哪些关键函数"。你需要:

1. 从题目文本里识别出触发异常的函数名(error_trigger)
2. 在上面的 tries 里找到包含这个 trigger 的 try 块
3. 构造 error_path = [root, trigger, ...清洗后的 except_calls]
4. 清洗规则:
   - 去除名字以 Error/Exception/Warning 结尾的条目(它们是异常类,不是处理函数)
   - 去除 isinstance/cast/len/print 等噪声
   - 保留 ToolMessage / 类实例化这种"构造错误响应"的关键步骤
   - 每条 except_calls 最多保留 4 个

输出 JSON:
{{
  "error_trigger": "触发异常的函数名",
  "error_path": ["{ctx.root_short}", "trigger", "handler1", "handler2", ...],
  "explanation": "2-4 句自然语言解释:什么情况下会抛异常、各处理函数的作用、整体错误处理策略"
}}
"""
    return SYSTEM_COMMON, user


# ============================================================
# 题型 5:core_support_classification
# ============================================================

def prompt_core_support(ctx: LocalContext, question: str) -> tuple[str, str]:
    user = f"""# 题目
{question}

# Root 函数
{ctx.root_short}

# Root 的主链(预计算)
{' → '.join(ctx.main_path)}

# 树形子图(带控制流标注,每个节点带角色提示)
{ctx.subgraph_tree}

# Callees 摘要(带启发式角色标签)
{_format_callees(ctx)}

# Root 源码(前 60 行,用于判断 callee 在 root 里的真实使用方式)
```python
{_format_source(ctx, max_lines=60)}
```

# 要求
把 root 函数的所有 callees 分成两类:

- **core_entities**(核心业务函数):推动主要业务流程的函数。
- **support_entities**(辅助函数):日志、格式化、类型检查、异常类、叶子工具函数。

判断原则(按优先级):
1. **如果 callee 的 out_degree >= 1(它还调用别的函数),强倾向 core** —— 这是最重要的规则
2. role_hint 是 leaf 的几乎都归 support(出度为 0,就是叶子工具)
3. role_hint 是 logger 的全部归 support
4. 名字明显是辅助动作的(_get_/_format_/_to_/_is_/_has_)归 support,即使 out_degree >= 1
5. 主链上的函数强倾向 core
6. 类实例化(大写开头):
   - 名字以 Error/Warning/Exception 结尾 → support(异常类)
   - 在主链上 → core
   - 否则看源码里它是怎么被用的:如果只是构造一个数据对象传给别人 → support;如果它本身是被作为目标产出 → core

输出 JSON(只用 short_name):
{{
  "core_entities": ["func1", "func2", ...],
  "support_entities": ["helper1", "helper2", ...],
  "explanation": "2-4 句自然语言解释:为什么这些是核心/辅助、它们各自的角色和协作关系"
}}
"""
    return SYSTEM_COMMON, user


# ============================================================
# 题型 6:local_subgraph_summary
# ============================================================

def prompt_subgraph_summary(ctx: LocalContext, question: str) -> tuple[str, str]:
    user = f"""# 题目
{question}

# Root 函数
{ctx.root_short}

# Root docstring
{ctx.root_docstring[:300] if ctx.root_docstring else '(无)'}

# 主链
{' → '.join(ctx.main_path)}

# 树形子图
{ctx.subgraph_tree}

# Callees 摘要
{_format_callees(ctx)}

# 要求
总结以 {ctx.root_short} 为根的局部调用子图整体完成了什么任务。

1. 把主链上的函数按"阶段"分类,可选的阶段标签是:
   - 输入解析 (parse/load/read/decode)
   - 合法性校验 (validate/check/verify/ensure)
   - 核心执行 (run/execute/process/invoke/forward/generate)
   - 输出构造 (build/format/render/return/write/encode)
2. 提取 2~3 个最核心的函数作为 core_entities
3. stages 按在主链上的出现顺序排列,去重

# 解释约束(非常重要)
- explanation 只能围绕你最终输出的 stages 和 core_entities 来写
- 不要新增阶段名; explanation 中出现的阶段必须完全来自 stages 字段
- 不要把 tree/callees 里其他函数升级为核心函数; explanation 中点名的核心函数必须来自 core_entities
- 不要额外引入“合法性校验”这类未出现在 stages 里的阶段
- 每个阶段最多用一句话解释其作用,优先写阶段之间如何衔接,不要扩展无关的实现细节

输出 JSON:
{{
  "stages": ["阶段1", "阶段2", ...],
  "core_entities": ["最关键的 1~3 个函数"],
  "explanation": "2-4 句自然语言解释:子图整体完成什么业务功能、各阶段如何衔接、核心函数的作用"
}}
"""
    return SYSTEM_COMMON, user


# ============================================================
# 题型 7:dependency_trace (跨函数追踪,纯调用图)
# ============================================================

def prompt_dependency_trace(
    ctx: LocalContext,
    question: str,
    descendants: list[dict],
    target_kind_hint: str = "",
) -> tuple[str, str]:
    """
    descendants 是 GraphRAGTools.get_khop_descendants 的输出。
    每项: {short_name, hop, role_hint}
    """
    # 把 descendants 按 hop 分组,直观展示给 LLM
    by_hop: dict[int, list[str]] = {}
    for d in descendants:
        path = d.get("sample_path")
        path_text = f" path={' -> '.join(path)}" if path else ""
        by_hop.setdefault(d["hop"], []).append(
            f"{d['short_name']} [{d['role_hint']}]{path_text}"
        )
    desc_text_lines = []
    for hop in sorted(by_hop.keys()):
        items = by_hop[hop][:30]  # 每跳最多展示 30 个
        more = f" ... (+{len(by_hop[hop]) - 30} more)" if len(by_hop[hop]) > 30 else ""
        desc_text_lines.append(f"  hop {hop}: {', '.join(items)}{more}")
    desc_text = "\n".join(desc_text_lines) if desc_text_lines else "  (无可达节点)"

    user = f"""# 题目
{question}

# Root 函数
{ctx.root_short}

# 通过调用图追踪到的可达节点(按距离 hop 分组)
{desc_text}

# 要求
题目会问"从 root 出发,经过 ≤ N 跳能到达哪些某类目标函数"。你需要:

1. 从题目里识别出"目标类别"(例如:类实例化 / 辅助函数 / Chat模型 / 异常处理函数 等)
2. 在上面的可达节点列表里,挑出符合该类别的所有节点
3. 同时给一条代表性最短路径,从 root 到一个典型目标节点

# 判断目标类别的提示
- "类实例化": 选择调用图或源代码明确表示的构造目标；Error/Exception/Warning 只有在存在明确构造证据时才排除,不能因为名称本身是异常类就排除
- "辅助函数": 短名以 _handle / _validate / _format / _check / _convert 等前缀
- 如果题目明确说了某个具体前缀(如 "Chat" 开头),严格按字面匹配

输出 JSON:
{{
  "target_kind": "类别标识符",
  "target_entities": ["entity1", "entity2", ...],
  "sample_path": ["{ctx.root_short}", "中间节点", "目标节点"],
  "explanation": "2-4 句自然语言解释:为什么能到达这些目标、路径上各步的作用、这些依赖的意义"
}}
"""
    return SYSTEM_COMMON, user


# ============================================================
# 题型 8:impact_scope (反向调用关系,纯调用图)
# ============================================================

def prompt_impact_scope(
    ctx: LocalContext,
    question: str,
    callers: dict[str, list[str]],
) -> tuple[str, str]:
    """
    callers 是 GraphRAGTools.get_reverse_callers 的输出:
      {"direct": [...], "indirect": [...]}
    """
    direct_text = ", ".join(callers["direct"][:20])
    if len(callers["direct"]) > 20:
        direct_text += f" ... (+{len(callers['direct']) - 20} more)"
    indirect_text = ", ".join(callers["indirect"][:30])
    if len(callers["indirect"]) > 30:
        indirect_text += f" ... (+{len(callers['indirect']) - 30} more)"

    user = f"""# 题目
{question}

# Root 函数
{ctx.root_short}

# 反向调用关系(从调用图反向 BFS 得到, 已过滤 pseudo/synthetic/test callers)
- 直接调用者(1 跳): {direct_text or '(无)'}
- 间接调用者(2 跳, 不含直接): {indirect_text or '(无)'}

# 要求
题目问的是当前静态调用图中的调用者范围。你的任务非常简单 —— 把上面的两组数据原样输出,
但要做这些清洗:

1. 去重(同一个 short_name 只保留一次)
2. 去掉明显的噪声(测试函数 test_xxx, __init__, __main__ 等)
3. 潜在影响范围最多保留 12 个(包含直接调用者和 2 跳调用者)
4. 不要引入上面 direct/indirect 列表之外的 caller
5. explanation 必须用英文写 2-4 句,并且必须覆盖下面三点:
   - The static direct callers are ...
   - The conservative potential impact points are ...
   - These potential impact points are static reverse-reachability candidates and are not guaranteed runtime effects or complete dynamic impact sets.
6. explanation 中必须原样包含短语 "potential impact points" 和 "not guaranteed runtime effects"

输出 JSON:
{{
  "static_direct_callers": [...],
  "potentially_affected_callers": [...],
  "explanation": "2-4 English sentences covering direct callers, conservative potential impact points, and the static-only caveat"
}}
"""
    return SYSTEM_COMMON, user


# ============================================================
# 分派器
# ============================================================

def build_prompt(
    qtype: str,
    ctx: LocalContext,
    question: str,
    tools,  # GraphRAGTools
) -> tuple[str, str]:
    """根据题型选择合适的 prompt 模板,组装 system+user 消息。"""
    # Deprecated dispatcher. The current GPT-5.5 SU8 pipeline builds local
    # prediction prompts via run_three_methods_clean_v1_en.py::build_prediction_prompt
    # and the per-type build_structure_schema; this dispatcher and the per-type
    # ``prompt_*`` helpers it calls are not used by that pipeline.
    if qtype == "execution_flow":
        return prompt_execution_flow(ctx, question)
    elif qtype == "result_origin":
        return prompt_result_origin(ctx, question)
    elif qtype == "conditional_path":
        branches = tools.get_branch_details(ctx.root_qualname)
        return prompt_conditional_path(ctx, question, branches)
    elif qtype == "error_handling_path":
        tries = tools.get_try_except_details(ctx.root_qualname)
        return prompt_error_handling(ctx, question, tries)
    elif qtype == "core_support_classification":
        return prompt_core_support(ctx, question)
    elif qtype == "local_subgraph_summary":
        return prompt_subgraph_summary(ctx, question)
    elif qtype == "dependency_trace":
        descendants = tools.get_khop_descendants(ctx.root_qualname, max_hops=3)
        return prompt_dependency_trace(ctx, question, descendants)
    elif qtype == "impact_scope":
        callers = tools.get_reverse_callers(ctx.root_qualname, max_hops=2)
        return prompt_impact_scope(ctx, question, callers)
    else:
        raise ValueError(f"unknown question type: {qtype}")


# ============================================================
# Ablation 用的 prompt 变体:Direct 和 Code-only
# ============================================================
#
# Direct:   完全不给上下文,只给问题 + JSON 输出格式要求
# Code-only: 只给 root 函数源码,没有 AST 派生、没有调用图、没有 tree
# Structured(就是上面 build_prompt 的版本):AST + 调用图 + tree + role_hint
#
# 三个变体的输出 schema 必须完全一致,才能用同一个 evaluator 打分。

# 每类题的 JSON schema 描述(放在 prompt 末尾让 LLM 知道要输出什么)
#
# Deprecated schema variant: ``_OUTPUT_SCHEMAS`` backs only the legacy ablation
# prompt builders below. The current GPT-5.5 SU8 pipeline does NOT use it; local
# predictions use run_three_methods_clean_v1_en.py::build_prediction_prompt
# together with build_structure_schema (the canonical per-type output schema).
_OUTPUT_SCHEMAS: dict[str, str] = {
    "execution_flow":
        '{"path_type": "representative_path", "path_condition": "", "path_nodes": ["root_short", "func2", "func3"], "coverage_note": "This is one extracted path, not the unique full execution flow.", "explanation": "2-4句自然语言解释"}',
    "result_origin":
        '{"target_object": "...", "source_entity": "...", "consumer_entities": [...], "explanation": "2-4句自然语言解释"}',
    "conditional_path":
        '{"condition": "...", "branch_kind": "if_true|if_false", "branch_span": {"start_line": 0, "end_line": 0}, "path_nodes": [...], "excluded_branches": [...], "explanation": "2-4句自然语言解释"}',
    "error_handling_path":
        '{"error_trigger": "...", "error_path": [...], "error_action": "...", "fallback_action": null, "evidence_span": {"start_line": 0, "end_line": 0}, "explanation": "2-4句自然语言解释"}',
    "core_support_classification":
        '{"core_entities": [...], "support_entities": [...], "explanation": "2-4句自然语言解释"}',
    "local_subgraph_summary":
        '{"stages": [...], "core_entities": [...], "explanation": "2-4句自然语言解释"}',
    "dependency_trace":
        '{"target_kind": "...", "target_entities": [...], "sample_path": [...], "explanation": "2-4句自然语言解释"}',
    "impact_scope":
        '{"static_direct_callers": [...], "potentially_affected_callers": [...], "explanation": "2-4句自然语言解释"}',
}


def build_direct_prompt(qtype: str, question: str, root_short: str) -> tuple[str, str]:
    """
    Direct baseline:不给任何上下文,只告诉 LLM 题目和输出格式。
    用来测 LLM 在零上下文下的表现 —— 这通常会很差,因为 LLM 不知道函数体里有什么。
    """
    schema = _OUTPUT_SCHEMAS.get(qtype, "{}")
    user = f"""# 题目
{question}

# 待分析的函数
{root_short}

# 任务
请回答上述题目。注意:你没有源码,只能根据函数名做推测。

# 输出格式(必须是合法 JSON)
{schema}

# 要求
- 严格按照上述 JSON schema 输出
- 函数名使用 short name(不含模块路径)
- 只输出 JSON,不要有 markdown 标记
- explanation 字段必须包含 2-4 句自然语言解释,说明"为什么"
"""
    return SYSTEM_COMMON, user


def build_code_only_prompt(
    qtype: str, question: str, ctx: LocalContext,
) -> tuple[str, str]:
    """
    Code-only baseline:只给 root 函数源码,不给 AST 派生(branches/tries 等)、
    不给调用图、不给 tree、不给 role_hint。

    用来测 LLM 在"裸源码"下的表现 —— 可以反映 LLM 自己解析源码的能力。
    """
    schema = _OUTPUT_SCHEMAS.get(qtype, "{}")
    src = _format_source(ctx, max_lines=80)
    user = f"""# 题目
{question}

# 函数 {ctx.root_short} 的源码
```python
{src}
```

# 任务
请仅根据上面的源码回答题目。

# 输出格式(必须是合法 JSON)
{schema}

# 要求
- 严格按照上述 JSON schema 输出
- 函数名使用 short name(不含模块路径)
- 过滤明显的噪声调用:isinstance, cast, super, len, print, dict/list/tuple/set 等
- 只输出 JSON,不要有 markdown 标记
- explanation 字段必须包含 2-4 句自然语言解释,说明"为什么"
"""
    return SYSTEM_COMMON, user


def build_prompt_for_method(
    method: str,
    qtype: str,
    ctx: LocalContext,
    question: str,
    tools,
) -> tuple[str, str]:
    """
    统一入口:按 method 选择 prompt 构建器。
    method ∈ {"direct", "code_only", "structured"}
    """
    # Deprecated entry point. The current GPT-5.5 SU8 pipeline uses
    # run_three_methods_clean_v1_en.py::build_prediction_prompt + build_structure_schema
    # instead. (Only ``_format_source`` from this module is still imported by the
    # active pipeline; the module as a whole is NOT deprecated.)
    if method == "direct":
        return build_direct_prompt(qtype, question, ctx.root_short)
    elif method == "code_only":
        return build_code_only_prompt(qtype, question, ctx)
    elif method == "structured":
        return build_prompt(qtype, ctx, question, tools)
    else:
        raise ValueError(f"unknown method: {method}")
