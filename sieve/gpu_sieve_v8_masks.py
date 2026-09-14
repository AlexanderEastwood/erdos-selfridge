"""Qualified four-filter masks; scalar v8 fallback outside the production wheel set."""
from __future__ import annotations
from typing import Any
import gpu_sieve_v8 as base
from es_word_masks import Variant
QUALIFIED = {397: [[211, 1], [139, 1], [83, 1], [2, 9], [59, 1], [103, 1], [137, 1], [41, 1], [3, 5], [37, 1], [29, 1], [101, 1], [31, 1], [67, 1], [5, 2], [7, 1], [19, 1], [199, 1]], 399: [[223, 1], [3, 6], [211, 1], [139, 1], [83, 1], [59, 1], [103, 1], [137, 1], [5, 4], [2, 9], [37, 1], [29, 1], [101, 1], [13, 1], [31, 1], [67, 1]]}
Plan = base.Plan
TUNABLES = base.TUNABLES

def __getattr__(name: str) -> Any:
    return getattr(base, name)

def GpuSieve(plan: Any) -> Any:
    native: Any = base.GpuSieve(plan)
    explicit = plan.tun.get('word_masks', 'auto')
    assert explicit in ('auto', 'force', 'off')
    wheel = [[p, t] for p, t in plan.wheel]
    eligible = (plan.k in QUALIFIED and wheel == QUALIFIED[plan.k]) or explicit == 'force'
    estimate = sum(plan.info[p][0] for p in plan.rest[:4]) * ((native.r0 + 31) // 32) * 4
    if explicit != 'off' and eligible and native.n_fast >= 4 and plan.radix and native.r0 <= 64 and estimate <= 2 << 20:
        result = Variant(native, 'mask4-global', 4, shared=False)
        result.mask_mode = 'mask4-global'
        return result
    native.mask_mode = 'scalar-v8'
    return native
