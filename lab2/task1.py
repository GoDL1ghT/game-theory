"""
Лабораторная работа по теории массового обслуживания.

Задача 1. Многоканальная СМО с отказами (модель Эрланга, M/M/n/0).

Расчёт выполнен по формулам, приведённым в учебном пособии
М. А. Плескунова «Теория массового обслуживания» (2022).

Исходные данные:
    α = 1.0  — среднее время обработки одной заявки, ч
    N = 60   — среднее число заявок в сутки
    n = 5    — заданное число каналов обслуживания

Требуется:
    1) найти минимальное число каналов, при котором относительная
       пропускная способность Q >= 95 %;
    2) решить задачу для n = 5 каналов;
    3) определить предельные вероятности состояний и характеристики
       обслуживания: вероятность отказа, абсолютную и относительную
       пропускную способность, среднее число занятых каналов,
       коэффициент загрузки каналов.

Соглашение о точности (по методическим указаниям):
    • промежуточные вычисления — 5 знаков после запятой;
    • окончательные ответы     — 3 знака после запятой.
"""

import math
import os
import matplotlib.pyplot as plt


# Константы точности вывода
PREC_MID = 5      # знаков для промежуточных значений
PREC_FIN = 3      # знаков для окончательных ответов


# ----------------------------------------------------------------------
# Базовые расчётные функции (формулы Эрланга для M/M/n/0)
# ----------------------------------------------------------------------

def state_probabilities(rho: float, n: int) -> list:
    """
    Возвращает список предельных вероятностей состояний системы
    [P_0, P_1, ..., P_n] для n-канальной СМО с отказами.

    Формулы (Плескунова, гл. «СМО с отказами»):
        P_0 = 1 / Σ_{k=0..n} (rho^k / k!)
        P_k = (rho^k / k!) * P_0,   k = 1..n
    где rho = λ / μ — приведённая интенсивность нагрузки.
    """
    # Сумма ряда в знаменателе для нормировки
    denom = sum((rho ** k) / math.factorial(k) for k in range(n + 1))
    P0 = 1.0 / denom
    # Возвращаем все P_k от k = 0 до k = n
    return [(rho ** k) / math.factorial(k) * P0 for k in range(n + 1)]


def erlang_b(rho: float, n: int) -> float:
    """
    Вероятность отказа по формуле Эрланга B:
        P_отк = (rho^n / n!) / Σ_{k=0..n} (rho^k / k!) = P_n
    """
    return state_probabilities(rho, n)[n]


def system_characteristics(lam: float, mu: float, n: int) -> dict:
    """
    Рассчитывает все характеристики n-канальной СМО с отказами.

    Параметры:
        lam — интенсивность входного потока заявок, заявок/час;
        mu  — интенсивность обслуживания одного канала, заявок/час;
        n   — число каналов.

    Возвращает словарь характеристик:
        rho      — приведённая интенсивность нагрузки;
        P        — список предельных вероятностей [P_0..P_n];
        P_otkaz  — вероятность отказа;
        Q        — относительная пропускная способность;
        A        — абсолютная пропускная способность, заявок/час;
        k_busy   — среднее число занятых каналов;
        k_load   — коэффициент загрузки каналов.
    """
    rho = lam / mu                              # ρ = λ/μ
    P = state_probabilities(rho, n)             # предельные вероятности состояний
    P_otkaz = P[n]                              # P_отк = P_n (все каналы заняты)
    Q = 1.0 - P_otkaz                           # Q = 1 − P_отк
    A = lam * Q                                 # A = λQ
    k_busy = A / mu                             # k̄ = A/μ = ρQ — среднее число занятых каналов
    k_load = k_busy / n                         # k_з = k̄/n — коэффициент загрузки каналов
    return {
        "rho": rho,
        "P": P,
        "P_otkaz": P_otkaz,
        "Q": Q,
        "A": A,
        "k_busy": k_busy,
        "k_load": k_load,
    }


def find_min_channels(lam: float, mu: float, target_Q: float = 0.95) -> int:
    """
    Поиск минимального числа каналов n, при котором Q >= target_Q.

    Q монотонно возрастает с ростом n, поэтому достаточно
    последовательного перебора n = 1, 2, 3, ...
    """
    rho = lam / mu
    n = 1
    while True:
        Q = 1.0 - erlang_b(rho, n)
        if Q >= target_Q:
            return n
        n += 1


# ----------------------------------------------------------------------
# Вспомогательные функции вывода
# ----------------------------------------------------------------------

def fmt_mid(x: float) -> str:
    """Форматирование числа для промежуточных таблиц (5 знаков)."""
    return f"{x:.{PREC_MID}f}"


def fmt_fin(x: float) -> str:
    """Форматирование числа для итоговых ответов (3 знака)."""
    return f"{x:.{PREC_FIN}f}"


def print_state_table(P: list) -> None:
    """Печать таблицы предельных вероятностей состояний (5 знаков)."""
    print(f"  {'Состояние':<11} | {'Вероятность':>12}")
    print(f"  {'-' * 11}-+-{'-' * 12}")
    for i, p in enumerate(P):
        print(f"  P_{i:<9} | {fmt_mid(p):>12}")
    print(f"  {'Σ P_k':<11} | {fmt_mid(sum(P)):>12}  (контроль: должна быть 1)")


def print_dependency_table(lam: float, mu: float, n_max: int, n_min: int) -> None:
    """Таблица зависимости характеристик СМО от числа каналов n."""
    print(f"\n  {'n':>3} | {'P_отк':>10} | {'Q':>10} | {'A, з/ч':>10} | "
          f"{'k̄':>10} | {'k_з':>10}")
    print(f"  {'-' * 3}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-"
          f"{'-' * 10}-+-{'-' * 10}")
    for n in range(1, n_max + 1):
        r = system_characteristics(lam, mu, n)
        mark = "  ← min" if n == n_min else ""
        print(f"  {n:>3} | {fmt_mid(r['P_otkaz']):>10} | "
              f"{fmt_mid(r['Q']):>10} | {fmt_mid(r['A']):>10} | "
              f"{fmt_mid(r['k_busy']):>10} | {fmt_mid(r['k_load']):>10}{mark}")


def print_final_report(label: str, lam: float, mu: float, n: int) -> None:
    """Итоговый отчёт о характеристиках СМО для заданного n (3 знака)."""
    print("\n" + "=" * 72)
    print(f"  {label}")
    print("=" * 72)
    res = system_characteristics(lam, mu, n)

    print(f"Число каналов обслуживания:             n = {n}")
    print(f"Приведённая интенсивность нагрузки:     ρ = {fmt_mid(res['rho'])}")
    print()
    print("Предельные вероятности состояний:")
    print_state_table(res["P"])
    print()
    print("ОКОНЧАТЕЛЬНЫЕ РЕЗУЛЬТАТЫ (округление до 3-го знака):")
    print(f"  Вероятность отказа                    P_отк = {fmt_fin(res['P_otkaz'])}")
    print(f"  Относительная пропускная способность  Q     = {fmt_fin(res['Q'])}")
    print(f"  Абсолютная пропускная способность     A     = {fmt_fin(res['A'])} заявок/час")
    print(f"                                              ≈ {fmt_fin(res['A'] * 24)} заявок/сутки")
    print(f"  Среднее число занятых каналов         k̄     = {fmt_fin(res['k_busy'])}")
    print(f"  Коэффициент загрузки каналов          k_з   = {fmt_fin(res['k_load'])} "
          f"({fmt_fin(res['k_load'] * 100)} %)")


# ----------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------

def plot_vs_channels(lam: float, mu: float, n_max: int, n_min: int,
                     target_Q: float, n_given: int, out_path: str) -> None:
    """
    График 1. Зависимости основных характеристик СМО от числа каналов n.

    Строятся две панели:
      • P_отк(n) и Q(n) с пороговой линией;
      • k̄(n) и k_з(n).
    """
    ns = list(range(1, n_max + 1))
    P_otk = [erlang_b(lam / mu, n) for n in ns]
    Q = [1 - p for p in P_otk]
    k_busy = [system_characteristics(lam, mu, n)["k_busy"] for n in ns]
    k_load = [system_characteristics(lam, mu, n)["k_load"] for n in ns]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ----- Левая панель: P_отк и Q -----
    ax = axes[0]
    ax.plot(ns, P_otk, "o-", color="crimson", label="P_отк(n)")
    ax.plot(ns, Q, "s-", color="seagreen", label="Q(n)")
    ax.axhline(target_Q, color="gray", linestyle="--",
               label=f"порог Q = {target_Q}")
    ax.axvline(n_min, color="gray", linestyle=":",
               label=f"n_min = {n_min}")
    ax.axvline(n_given, color="orange", linestyle=":",
               label=f"n = {n_given} (задано)")
    ax.set_xlabel("Число каналов n")
    ax.set_ylabel("Вероятность")
    ax.set_title("Вероятность отказа и относительная\nпропускная способность")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=9)

    # ----- Правая панель: k̄ и k_з -----
    ax = axes[1]
    ax.plot(ns, k_busy, "o-", color="steelblue", label="k̄(n) — занятые каналы")
    ax.plot(ns, k_load, "s-", color="darkorange", label="k_з(n) — загрузка")
    ax.axvline(n_min, color="gray", linestyle=":",
               label=f"n_min = {n_min}")
    ax.axvline(n_given, color="orange", linestyle=":",
               label=f"n = {n_given} (задано)")
    ax.set_xlabel("Число каналов n")
    ax.set_ylabel("Значение")
    ax.set_title("Среднее число занятых каналов\nи коэффициент загрузки")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=9)

    fig.suptitle("Зависимости характеристик СМО от числа каналов n  (ρ = λ/μ = "
                 f"{lam / mu:.2f})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_vs_lambda(mu: float, n: int, out_path: str) -> None:
    """
    График 2. Зависимости показателей СМО от интенсивности входного потока λ
    при фиксированном числе каналов n (показывает, как насыщается система
    при увеличении нагрузки).
    """
    lambdas = [0.1 * i for i in range(1, 81)]   # λ от 0.1 до 8 заявок/час
    P_otk, Q, A, k_load = [], [], [], []
    for lam in lambdas:
        r = system_characteristics(lam, mu, n)
        P_otk.append(r["P_otkaz"])
        Q.append(r["Q"])
        A.append(r["A"])
        k_load.append(r["k_load"])

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    # Левая ось — вероятности и коэффициент загрузки
    ax1.plot(lambdas, P_otk, color="crimson", label="P_отк(λ)")
    ax1.plot(lambdas, Q, color="seagreen", label="Q(λ)")
    ax1.plot(lambdas, k_load, color="darkorange", label="k_з(λ) — загрузка")
    ax1.set_xlabel("Интенсивность входного потока λ, заявок/час")
    ax1.set_ylabel("Вероятность / коэффициент загрузки")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Правая ось — абсолютная пропускная способность
    ax2 = ax1.twinx()
    ax2.plot(lambdas, A, color="steelblue", linestyle="--",
             label="A(λ), заявок/час")
    ax2.set_ylabel("Абсолютная пропускная способность A, заявок/час")

    # Объединённая легенда
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right",
               fontsize=9)

    ax1.set_title(f"Зависимости характеристик СМО от интенсивности потока λ "
                  f"при n = {n}, μ = {mu}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Главная программа
# ----------------------------------------------------------------------

def main() -> None:
    # ---------- Исходные данные ----------
    alpha = 1.0           # α — среднее время обработки одной заявки, ч
    N_day = 60            # N — среднее число заявок в сутки
    n_given = 5           # n — заданное число каналов
    target_Q = 0.95       # требуемая относительная пропускная способность

    # ---------- Приведение к одним временным единицам (час) ----------
    lam = N_day / 24      # λ — интенсивность потока, заявок/час
    mu = 1.0 / alpha      # μ — интенсивность обслуживания канала, заявок/час
    rho = lam / mu        # ρ — приведённая интенсивность нагрузки

    print("ИСХОДНЫЕ ДАННЫЕ И РАСЧЁТНЫЕ ПАРАМЕТРЫ")
    print("-" * 72)
    print(f"  Среднее время обслуживания заявки:    α = {alpha} ч")
    print(f"  Среднее число заявок в сутки:         N = {N_day}")
    print(f"  Заданное число каналов:               n = {n_given}")
    print()
    print(f"  Интенсивность поступления заявок:     λ = N/24 = {fmt_mid(lam)} заявок/час")
    print(f"  Интенсивность обслуживания канала:    μ = 1/α  = {fmt_mid(mu)} заявок/час")
    print(f"  Приведённая интенсивность нагрузки:   ρ = λ/μ  = {fmt_mid(rho)}")

    # ---------- Задание 1. Поиск минимального n для Q >= 95 % ----------
    n_min = find_min_channels(lam, mu, target_Q)

    print("\n" + "=" * 72)
    print(f"  ЗАДАНИЕ 1. Минимальное число каналов для Q ≥ {target_Q * 100:.0f} %")
    print("=" * 72)

    # Таблица зависимости характеристик от n
    n_max = max(n_given, n_min) + 3
    print("\nТаблица 1. Характеристики СМО при разном числе каналов "
          f"(ρ = {fmt_mid(rho)}):")
    print_dependency_table(lam, mu, n_max, n_min)

    print(f"\nОТВЕТ (зад. 1): минимальное число каналов n_min = {n_min}")

    # ---------- Задания 2 и 3. Расчёт для заданного n = 5 ----------
    print_final_report(
        f"ЗАДАНИЯ 2–3. Расчёт характеристик СМО для заданного n = {n_given}",
        lam, mu, n_given,
    )

    # Дополнительно: те же характеристики для n_min (нужно для выводов)
    if n_min != n_given:
        print_final_report(
            f"СПРАВОЧНО. Характеристики СМО при n = n_min = {n_min}",
            lam, mu, n_min,
        )

    # ---------- Графики ----------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    g1 = os.path.join(out_dir, "graph_vs_channels.png")
    g2 = os.path.join(out_dir, "graph_vs_lambda.png")
    plot_vs_channels(lam, mu, n_max, n_min, target_Q, n_given, g1)
    plot_vs_lambda(mu, n_given, g2)
    print(f"\nГрафики сохранены:\n  • {g1}\n  • {g2}")

    # ---------- Развёрнутые выводы ----------
    res5 = system_characteristics(lam, mu, n_given)
    res_min = system_characteristics(lam, mu, n_min)

    print("\n" + "=" * 72)
    print("  ВЫВОДЫ")
    print("=" * 72)
    print(
        f"1. При n = {n_given} каналах относительная пропускная способность "
        f"Q = {fmt_fin(res5['Q'])} ({fmt_fin(res5['Q'] * 100)} %), "
        f"вероятность отказа P_отк = {fmt_fin(res5['P_otkaz'])} "
        f"({fmt_fin(res5['P_otkaz'] * 100)} %). Условие Q ≥ "
        f"{target_Q * 100:.0f} % НЕ выполняется."
    )
    print(
        f"2. Для достижения требования Q ≥ {target_Q * 100:.0f} % достаточно "
        f"n_min = {n_min} каналов: при этом Q = {fmt_fin(res_min['Q'])}, "
        f"P_отк = {fmt_fin(res_min['P_otkaz'])}, абсолютная пропускная "
        f"способность возрастает с A = {fmt_fin(res5['A'])} до "
        f"A = {fmt_fin(res_min['A'])} заявок/час."
    )
    print(
        f"3. С увеличением числа каналов n абсолютная пропускная способность A "
        f"и относительная Q растут, а вероятность отказа P_отк убывает по "
        f"закону Эрланга. Однако коэффициент загрузки k_з падает: при n = "
        f"{n_given} он составляет {fmt_fin(res5['k_load'])} "
        f"({fmt_fin(res5['k_load'] * 100)} %), а при n = {n_min} — лишь "
        f"{fmt_fin(res_min['k_load'])} ({fmt_fin(res_min['k_load'] * 100)} %). "
        f"То есть выигрыш в пропускной способности достигается ценой простоя "
        f"каждого отдельного канала."
    )
    print(
        f"4. Влияние других параметров (см. график graph_vs_lambda.png): "
        f"при увеличении интенсивности λ при фиксированном n коэффициент "
        f"загрузки k_з и вероятность отказа P_отк растут, относительная "
        f"пропускная способность Q падает; абсолютная пропускная способность "
        f"A асимптотически стремится к n·μ = {n_given * mu} заявок/час "
        f"(физический предел системы). Уменьшение среднего времени "
        f"обслуживания α (рост μ) эквивалентно снижению ρ и улучшает все "
        f"показатели."
    )


if __name__ == "__main__":
    main()
