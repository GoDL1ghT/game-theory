"""Точка входа CLI: загрузка сценария, решение ЛП, сравнение с baseline.

Запуск:
    python main.py --config config/scenario.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from core.balancer import solve_min_max
from core.baselines import convex_optimal, equal_split
from core.evaluate import evaluate_allocation
from core.models import Scenario

console = Console()


def load_scenario(path: str) -> Scenario:
    """Читает YAML и валидирует его через Pydantic-схему :class:`Scenario`."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Scenario(**raw)


def _alloc_table(title: str, scenario: Scenario, lambdas: list[float]) -> Table:
    rep = evaluate_allocation(scenario.names, scenario.mus, lambdas)
    table = Table(title=title)
    table.add_column("Сервер", justify="left")
    table.add_column("μ", justify="right")
    table.add_column("λ", justify="right")
    table.add_column("ρ", justify="right")
    table.add_column("W, мс", justify="right")
    table.add_column("Устойч.", justify="center")
    for i, name in enumerate(scenario.names):
        m = rep.per_server[i]
        w_ms = "∞" if m.w == float("inf") else f"{m.w * 1000:.1f}"
        table.add_row(
            name, f"{scenario.mus[i]:.0f}", f"{lambdas[i]:.2f}",
            f"{m.rho:.3f}", w_ms, "✓" if m.stable else "✗",
        )
    return table, rep


def run(scenario: Scenario) -> None:
    """Прогоняет ЛП-распределение и эталоны, печатает сравнительные таблицы."""
    console.print("[bold green]ServerBalancer — распределение нагрузки[/bold green]\n")
    console.print(
        f"Серверов: {len(scenario.servers)} | "
        f"Σμ = {sum(scenario.mus):.0f} запр./с | "
        f"Λ = {scenario.total_lambda:.0f} запр./с | rho_max = {scenario.rho_max}\n"
    )

    # 1. ЛП min-max
    alloc = solve_min_max(scenario.mus, scenario.total_lambda, scenario.rho_max)
    if not alloc.feasible:
        console.print(f"[red]ЛП не решено: {alloc.status} (пул перегружен)[/red]")
        return
    lp_table, lp_rep = _alloc_table("ЛП (min-max загрузки)", scenario, alloc.lambdas)

    # 2. Эталон: равномерный сплит
    eq_lambdas = equal_split(scenario.mus, scenario.total_lambda)
    eq_table, eq_rep = _alloc_table("Baseline: равномерный сплит", scenario, eq_lambdas)

    # 3. Теоретический оптимум по W (выпуклая задача)
    cv_lambdas = convex_optimal(scenario.mus, scenario.total_lambda, scenario.rho_max)
    cv_rep = evaluate_allocation(scenario.names, scenario.mus, cv_lambdas)

    console.print(lp_table)
    console.print(eq_table)

    # Итоговое сравнение
    summary = Table(title="Сравнение стратегий (среднее по системе)")
    summary.add_column("Стратегия", justify="left")
    summary.add_column("Сред. W, мс", justify="right")
    summary.add_column("Сред. P(ожид.)", justify="right")
    summary.add_column("max ρ", justify="right")
    for label, rep in (
        ("Равномерный сплит", eq_rep),
        ("ЛП (min-max)", lp_rep),
        ("Оптимум по W (выпукл.)", cv_rep),
    ):
        w_ms = "∞" if rep.avg_w == float("inf") else f"{rep.avg_w * 1000:.1f}"
        summary.add_row(label, w_ms, f"{rep.avg_p_wait:.3f}", f"{rep.max_rho:.3f}")
    console.print(summary)

    # Проверка SLA для ЛП-решения
    sla_w = lp_rep.avg_w * 1000 <= scenario.sla_w_ms
    status = "[green]соблюдён[/green]" if sla_w else "[red]нарушен[/red]"
    console.print(
        f"\nSLA по времени отклика (≤ {scenario.sla_w_ms:.0f} мс): {status} "
        f"(ЛП даёт {lp_rep.avg_w * 1000:.1f} мс)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ServerBalancer CLI")
    parser.add_argument("--config", default="config/scenario.yaml",
                        help="Путь к YAML-сценарию")
    args = parser.parse_args()

    if not Path(args.config).exists():
        console.print(f"[red]Файл конфигурации не найден: {args.config}[/red]")
        return 1
    try:
        scenario = load_scenario(args.config)
        run(scenario)
    except Exception as exc:  # noqa: BLE001 — единая точка вывода ошибок CLI
        console.print(f"[red]Ошибка выполнения: {exc}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
