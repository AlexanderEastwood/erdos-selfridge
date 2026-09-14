"""Regenerate publication figures from the exact decimal table."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ref.erdos_ref import digits, primes_upto


def log_expectation(k: int) -> float:
    return sum(math.log10(p / (p - d)) for p in primes_upto(k) for d in digits(k, p))


def main() -> None:
    data: dict[str, str] = json.loads((ROOT / 'data/terms.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for ax, start in zip(axes, (1, 356)):
        previous = list(range(start, 378))
        current = list(range(378, 401))
        domain = list(range(start, 401))
        ax.plot(domain, [log_expectation(k) for k in domain], color='#cf823e', alpha=.7,
                linewidth=1, label='Density heuristic: log₁₀(Mₖ/Rₖ)')
        ax.scatter(previous, [math.log10(int(data[str(k)])) for k in previous],
                   color='#89939a', s=12, label='Previously published: 1–377', zorder=3)
        ax.scatter(current, [math.log10(int(data[str(k)])) for k in current],
                   color='#146b57', s=23, label='This computation: 378–400', zorder=4)
        ax.set_xlabel('k')
        ax.set_ylabel('log₁₀ g(k)')
        ax.set_title('All 400 values' if start == 1 else 'Frontier detail: k = 356–400')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(loc='upper left', fontsize=8, frameon=False)
    for extension in ('png', 'pdf'):
        fig.savefig(ROOT / f'paper/terms_growth.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
