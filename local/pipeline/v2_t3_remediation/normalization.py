"""Shared entity normalization for the benchmark generator and evaluator.

T3 remediation (author-approved, T3.6 section 6.9): the evaluator and the
generator must use the SAME tested normalization rule.  The v1 rule
(``x.rsplit(".", 1)[-1].strip().lower()``) mangled C++ qualified /
template-specialization names (``ggml::cpu::repack::gemm``,
``tensor_traits<...>::forward_mul_mat``), making source-correct gold
unscoreable (T3 cluster C-IMPACT-NORM).

The v2 rule keeps v1 behavior for every name that does not contain ``::``
or a template argument list, and additionally:

1. strips ``<...>`` template argument lists (but never when the whole
   string is an ``<anon@file:line>`` pseudo-name);
2. splits on ``::`` as well as ``.`` and takes the last segment;
3. lowercases and strips, exactly like v1.

``norm_v1`` is kept for drift reporting and regression tests.
"""

from __future__ import annotations

import re

_TEMPLATE_ARG_RE = re.compile(r"<[^>]*>")


def norm_v1(x: str) -> str:
    """The frozen v1 evaluator normalization (do not change)."""
    return x.rsplit(".", 1)[-1].strip().lower()


def _strip_template_args(x: str) -> str:
    stripped = _TEMPLATE_ARG_RE.sub("", x)
    # '<anon@file:line>' pseudo-names consist solely of a bracketed token;
    # stripping would erase the whole name, so keep the original.
    return stripped if stripped.strip() else x


def norm(x: str) -> str:
    """Unified v2 normalization used by generator and evaluator alike."""
    y = _strip_template_args(x.strip())
    y = y.replace("::", ".")
    return y.rsplit(".", 1)[-1].strip().lower()


def drift(x: str) -> bool:
    """True when the v2 rule normalizes ``x`` differently from v1."""
    return norm(x) != norm_v1(x)
