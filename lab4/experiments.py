"""Эксперименты ServerBalancer: сравнение ЛП с baseline на нескольких сценариях.

Генерирует графики в docs/figures/ и сводную таблицу docs/results.csv.
CSV сохраняется в формате, дружественном к русскому Excel: разделитель «;»,
десятичная запятая, кодировка UTF-8 с BOM.

Запуск: ``python experiments.py``
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.balancer import solve_min_max
from core.baselines import convex_optimal, equal_split
from core.evaluate import evaluate_allocation

FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MUS = [50.0, 30.0, 20.0, 80.0]
NAMES = ["web-1", "web-2", "web-3", "web-4"]
SUM_MU = sum(MUS)

SCENARIOS = {
    "Штатный (Λ=120)": 120.0,
    "Пограничный (Λ=72)": 72.0,
    "Стрессовый (Λ=160)": 160.0,
}


def evaluate_all(total_lambda: float):
    """Возвращает отчёты для трёх стратегий при заданном потоке."""
    lp = solve_min_max(MUS, total_lambda, rho_max=0.95)
    lp_rep = evaluate_allocation(NAMES, MUS, lp.lambdas) if lp.feasible else None
    eq_rep = evaluate_allocation(NAMES, MUS, equal_split(MUS, total_lambda))
    cv_rep = evaluate_allocation(NAMES, MUS, convex_optimal(MUS, total_lambda, 0.95))
    return lp, lp_rep, eq_rep, cv_rep


# --- Форматирование для русского Excel (десятичная запятая) ---------------

def _w(rep) -> str:
    """Среднее время отклика в мс с запятой; ∞ для неустойчивой системы."""
    v = rep.avg_w * 1000.0
    return "∞" if math.isinf(v) else f"{v:.1f}".replace(".", ",")


def _rho(rep) -> str:
    return f"{rep.max_rho:.3f}".replace(".", ",")


def _yn(flag: bool) -> str:
    return "да" if flag else "нет"


def build_rows() -> list[list[str]]:
    """Строки сводной таблицы по всем сценариям и стратегиям."""
    rows = []
    for label, lam in SCENARIOS.items():
        _, lp_rep, eq_rep, cv_rep = evaluate_all(lam)
        rows.append([
            label, f"{lam:.0f}",
            _w(eq_rep), _rho(eq_rep), _yn(eq_rep.all_stable),
            _w(lp_rep), _rho(lp_rep), _yn(lp_rep.all_stable),
            _w(cv_rep), _rho(cv_rep),
        ])
    return rows


HEADER = [
    "Сценарий", "Λ, запр/с",
    "Равном. W, мс", "Равном. max ρ", "Равном. устойчив",
    "ЛП W, мс", "ЛП max ρ", "ЛП устойчив",
    "Выпукл. W, мс", "Выпукл. max ρ",
]


def write_csv(path: str = "docs/results.csv") -> None:
    """Пишет CSV для русского Excel: «;»-разделитель, запятая, UTF-8 BOM."""
    rows = build_rows()
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(HEADER)
        writer.writerows(rows)
    return rows


# --- Графики --------------------------------------------------------------

def plot_allocation(total_lambda: float = 120.0) -> None:
    """Bar chart: распределение потока λ по серверам (равномерный vs ЛП)."""
    _, lp_rep, eq_rep, _ = evaluate_all(total_lambda)
    x = np.arange(len(NAMES))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - width / 2, eq_rep.lambdas, width, label="Равномерный сплит", color="#d98880")
    ax.bar(x + width / 2, lp_rep.lambdas, width, label="ЛП (min-max)", color="#5499c7")
    for i, mu in enumerate(MUS):
        ax.hlines(mu, i - 0.45, i + 0.45, colors="#566573", linestyles="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nμ={int(m)}" for n, m in zip(NAMES, MUS)])
    ax.set_ylabel("Поток λ, запр./с")
    ax.set_title(f"Распределение трафика по серверам (Λ={int(total_lambda)})\n"
                 "пунктир — мощность μ сервера")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_allocation.png", dpi=130)
    plt.close(fig)


def plot_utilization(total_lambda: float = 120.0) -> None:
    """Bar chart: загрузка ρ по серверам (равномерный vs ЛП), линия ρ=1."""
    _, lp_rep, eq_rep, _ = evaluate_all(total_lambda)
    eq_rho = [min(r, 2.0) for r in (l / m for l, m in zip(eq_rep.lambdas, MUS))]
    lp_rho = [l / m for l, m in zip(lp_rep.lambdas, MUS)]
    x = np.arange(len(NAMES))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - width / 2, eq_rho, width, label="Равномерный сплит", color="#d98880")
    ax.bar(x + width / 2, lp_rho, width, label="ЛП (min-max)", color="#5499c7")
    ax.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.2, label="ρ=1 (предел)")
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("Загрузка ρ")
    ax.set_title(f"Загрузка серверов (Λ={int(total_lambda)})\n"
                 "равномерный сплит перегружает медленные узлы")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_utilization.png", dpi=130)
    plt.close(fig)


def plot_load_sweep() -> None:
    """Кривая среднего времени отклика от суммарной нагрузки Λ."""
    capacity = 0.95 * SUM_MU
    lams = np.linspace(5, capacity - 1, 60)
    lp_w, eq_w, cv_w = [], [], []
    for lam in lams:
        _, lp_rep, eq_rep, cv_rep = evaluate_all(float(lam))
        lp_w.append(lp_rep.avg_w * 1000)
        cv_w.append(cv_rep.avg_w * 1000)
        eq_w.append(eq_rep.avg_w * 1000 if eq_rep.all_stable else np.nan)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(lams, eq_w, label="Равномерный сплит", color="#d98880", linewidth=2)
    ax.plot(lams, lp_w, label="ЛП (min-max)", color="#5499c7", linewidth=2)
    ax.plot(lams, cv_w, label="Оптимум по W (выпукл.)", color="#52be80",
            linewidth=2, linestyle="--")
    ax.axvline(4 * min(MUS), color="#d98880", linestyle=":", linewidth=1,
               label="порог устойчивости равномерного")
    ax.set_xlabel("Суммарный поток Λ, запр./с")
    ax.set_ylabel("Среднее время отклика W, мс")
    ax.set_title("Время отклика vs нагрузка\n(равномерный сплит расходится у Λ=80)")
    ax.set_ylim(0, 400)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_load_sweep.png", dpi=130)
    plt.close(fig)


def main() -> None:
    rows = write_csv()
    plot_allocation(120.0)
    plot_utilization(120.0)
    plot_load_sweep()

    print("Сводка по сценариям:")
    for r in rows:
        print(f"  {r[0]:22s} | равном. W={r[2]:>6s} мс (устойч.={r[4]}) "
              f"| ЛП W={r[5]:>5s} мс (maxρ={r[6]}) | выпукл. W={r[8]:>5s} мс")
    print("\nГрафики сохранены в docs/figures/, таблица — docs/results.csv (UTF-8 BOM, ;)")


if __name__ == "__main__":
    main()
