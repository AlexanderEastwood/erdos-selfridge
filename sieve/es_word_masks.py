"""Exact CRT word-mask prototype, retaining original candidate IDs and v8 late filters."""
from __future__ import annotations
import importlib
import importlib.util
from pathlib import Path
import sys
import time
from typing import Any, cast

import numpy as np

HERE = Path(__file__).resolve().parent


import gpu_sieve_v8 as ES


def phase_tables(plan: Any, count: int, terms: list[int]) -> tuple[np.ndarray, list[int]]:
    width = (len(terms) + 31) // 32
    pieces: list[np.ndarray] = []
    offsets: list[int] = []
    current = 0
    for prime in plan.rest[:count]:
        q, allowed = plan.info[prime]
        flags = np.zeros(q, dtype=np.bool_)
        flags[allowed] = True
        phases = np.arange(q, dtype=np.int64)
        table = np.zeros((q, width), dtype=np.uint32)
        for index, term in enumerate(terms):
            accepted = flags[(phases + term % q) % q].astype(np.uint32)
            table[:, index // 32] |= accepted << np.uint32(index % 32)
        offsets.append(current)
        flat = table.reshape(-1)
        pieces.append(flat)
        current += len(flat)
    return (np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.uint32)), offsets


def source_for(base: Any, count: int, offsets: list[int], shared_words: int, name: str) -> str:
    source = base.src.replace("void es_sieve(", "void " + name + "(")
    source = source.replace("const unsigned int out_cap)", "const unsigned int out_cap, const unsigned int* __restrict__ masks, const unsigned int* __restrict__ inner_ids)")
    # Scalar filters keep v8's original T0Q ordering; only the term array is sorted.
    source = source.replace("u + d0]", "u + original_d]")
    source = source.replace("+ (unsigned long long)d0;", "+ (unsigned long long)original_d;")
    loop = f"        for (unsigned int d0 = 0u; d0 < {base.r0}u; ++d0) {{"
    assert source.count(loop) == 1
    if count == 0:
        return source.replace(loop, loop + "\n            unsigned int original_d=inner_ids[d0];")
    if shared_words:
        staging = f"    __shared__ unsigned int wm[{shared_words}u];\n    for(unsigned int z=threadIdx.x;z<{shared_words}u;z+=blockDim.x)wm[z]=masks[z];\n"
        barrier = "    __syncthreads();"
        if barrier in source:
            source = source.replace(barrier, staging + barrier, 1)
        else:
            marker = "    unsigned long long gid ="
            source = source.replace(marker, staging + barrier + "\n" + marker, 1)
    words = (base.r0 + 31) // 32
    tail = (1 << (base.r0 % 32)) - 1 if base.r0 % 32 else 2**32 - 1
    code = [
        "        unsigned long long threshold_lo=M_LO-b_lo;",
        "        unsigned long long threshold_hi=M_HI-b_hi-(M_LO<b_lo);",
        f"        unsigned int left=0u,right={base.r0}u;",
        "        while(left<right){unsigned int mid=(left+right)/2u;",
        "          unsigned long long hi=term_hi[mid],lo=term_lo[mid];",
        "          if(hi<threshold_hi || (hi==threshold_hi && lo<threshold_lo))left=mid+1u;else right=mid;}",
        "        const unsigned int split=left;",
        "        #pragma unroll",
        f"        for(unsigned int word=0u;word<{words}u;++word){{",
        f"            unsigned int valid=(word=={words-1}u)?{tail}u:0xffffffffu;",
        "            unsigned int carry=split<=word*32u?valid:(split>=word*32u+32u?0u:((0xffffffffu<<(split-word*32u))&valid));",
        "            unsigned int bits=valid;",
    ]
    for j in range(count):
        q, off = base.rest_q[j], offsets[j]
        memory = "wm" if off + q * words <= shared_words else "masks"
        correction = (q - base.plan.M % q) % q
        code += [f"            {{unsigned int phase=BQ{j}+{correction}u;if(phase>={q}u)phase-={q}u;",
                 "             unsigned int keep;",
                 f"             if(!carry)keep={memory}[{off}u+BQ{j}*{words}u+word];",
                 f"             else if(carry==valid)keep={memory}[{off}u+phase*{words}u+word];",
                 f"             else keep=({memory}[{off}u+BQ{j}*{words}u+word]&~carry)|({memory}[{off}u+phase*{words}u+word]&carry);",
                 "             bits&=keep;}", "            if(!bits)continue;"]
        old = f"            {{ unsigned int m = BQ{j} + t0q["
        first = source.index(old)
        end = source.index("            if (!ok) continue;", first) + len("            if (!ok) continue;")
        source = source[:first] + source[end:]
    code += ["            while(bits){", "              unsigned int d0=word*32u+__ffs(bits)-1u;bits&=bits-1u;",
             "              unsigned int original_d=inner_ids[d0];"]
    source = source.replace(loop, "\n".join(code))
    ending = "        }\n    }\n}"
    assert source.endswith(ending)
    return source[:-len(ending)] + "            }\n        }\n    }\n}"


class Variant:
    def __init__(self, base: Any, name: str, filters: int = 0, shared: bool = True) -> None:
        cp: Any = importlib.import_module("cupy")
        self.base = base
        self.name = name
        self.mask_mode: str = name
        self.filters = filters
        self.cp = cp
        self.launches = 0
        self.residues_checked = 0
        self.out_cap: int = base.out_cap
        self.out: Any = base.out
        self.ochunk: int = base.ochunk
        self.extra: tuple[Any, ...] = ()
        self.setup_seconds = 0.0
        self.table_bytes = 0
        self.mask_shared_bytes = 0
        self.kernel = base.kernel
        self.term_lo, self.term_hi = base.term_lo, base.term_hi
        if name == "native":
            self.source = base.src
            return
        began = time.perf_counter()
        assert 0 <= filters <= base.n_fast and base.r0 > 0
        original = [(r * base.plan.basis[0]) % base.plan.M for r in base.plan.rings[0][1]]
        order = sorted(range(base.r0), key=lambda i: original[i])
        terms = [original[i] for i in order]
        lo, hi = base.term_lo.get(), base.term_hi.get()
        lo[:base.r0] = [v & (2**64 - 1) for v in terms]
        hi[:base.r0] = [v >> 64 for v in terms]
        self.term_lo, self.term_hi = cp.asarray(lo), cp.asarray(hi)
        packed, offsets = phase_tables(base.plan, filters, terms)
        shared_words = 0
        if shared:
            budget = max(0, 11264 - base.smem_words)
            for j in range(filters):
                end = offsets[j] + base.rest_q[j] * ((base.r0 + 31) // 32)
                if end <= budget:
                    shared_words = end
                else:
                    break
        self.table_bytes = int(packed.nbytes) if filters else 0
        self.mask_shared_bytes = shared_words * 4
        self.extra = (cp.asarray(packed), cp.asarray(order, dtype=cp.uint32))
        kernel_name = "es_" + name.replace("-", "_")
        self.source = source_for(base, filters, offsets, shared_words, kernel_name)
        self.kernel = cp.RawKernel(self.source, kernel_name, options=("-std=c++17",))
        self.kernel.compile()
        cp.cuda.get_current_stream().synchronize()
        self.setup_seconds = time.perf_counter() - began

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def _launch(self, start: int, count: int) -> None:
        blocks = min(65535 * 16, (count + self.threads - 1) // self.threads)
        self.kernel((blocks,), (self.threads,), (np.uint64(start), np.uint64(count), self.term_lo, self.term_hi,
                    self.tconst, self.t0q, self.allowed, self.out, self.out_n, np.uint32(self.out_cap), *self.extra))
        self.launches += 1

    def scan_block(self, block: int) -> list[int]:
        return ES.GpuSieve.scan_block(cast(Any, self), block)

    def _scan_block_chunked(self, block: int) -> list[int]:
        return ES.GpuSieve._scan_block_chunked(cast(Any, self), block)

    def search(self, max_blocks: int | None = None) -> dict[str, Any]:
        return ES.GpuSieve.search(cast(Any, self), max_blocks=max_blocks)


def build(plan: Any) -> tuple[Any, dict[str, Variant], float]:
    began = time.perf_counter()
    base = ES.GpuSieve(plan)
    baseline_setup = time.perf_counter() - began
    variants = {"native": Variant(base, "native"), "sorted-native": Variant(base, "sorted-native")}
    for count in (1, 2, 4):
        if count <= base.n_fast:
            variants[f"mask{count}"] = Variant(base, f"mask{count}", count)
    if base.n_fast >= 4:
        variants["mask4-global"] = Variant(base, "mask4-global", 4, shared=False)
    return base, variants, baseline_setup
