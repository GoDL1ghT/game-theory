"""
Лабораторная работа по теории массового обслуживания.

Задача 2. Многоканальная СМО с неограниченной очередью (M/M/k/∞).

Расчёт по формулам учебного пособия
М. А. Плескунова «Теория массового обслуживания» (2022).

Исходные данные:
    λ = 88   — интенсивность входного потока, заявок/сутки
    t = 2    — среднее время обслуживания одной заявки, минут
    a = 6    — коэффициент стоимости пребывания в очереди
    n = 4    — порог в расчёте вероятности «в очереди ≤ n заявок»

Требуется:
    1) Найти минимальное k_min при условии стационарности ρ/k < 1.
       Рассчитать при k_min: предельные вероятности состояний,
       L_оч и L_сист, T_оч и T_сист.
    2) Найти оптимальное k_opt, минимизирующее относительную
       стоимость C(k) = k/λ + a·T_оч.
    3) Сравнить характеристики при k_min и k_opt.
    4) Вычислить P(в очереди ≤ n заявок).

Соглашение о точности (по методическим указаниям):
    • промежуточные вычисления — 5 знаков после запятой;
    • окончательные ответы     — 3 знака после запятой;
    • ε = 1e-5 — оценка остатка ряда для бесконечного числа состояний.

Замечание о формуле стоимости. В условии формула C(k) = k/λ + a·T̂
содержит опечатку («Tобез»). По смыслу («издержки от пребывания
заявок В ОЧЕРЕДИ») и по эквивалентной классической форме
C_отн = k + a·L_оч (последняя через формулу Литтла даёт точно
k/λ + a·T_оч после деления на λ), используем T̂ = T_оч.
"""

import math
import os
import matplotlib.pyplot as plt


# Константы точности и сходимости
PREC_MID = 5       # знаков для промежуточных значений
PREC_FIN = 3       # знаков для окончательных ответов
EPS = 1e-5         # точность для оценки остатка ряда


def fmt_mid(x: float) -> str:
    """Промежуточное число (5 знаков)."""
    return f"{x:.{PREC_MID}f}"


def fmt_fin(x: float) -> str:
    """Окончательный ответ (3 знака)."""
    return f"{x:.{PREC_FIN}f}"


# ----------------------------------------------------------------------
# Расчётные функции (формулы СМО M/M/k/∞ из пособия Плескуновой)
# ----------------------------------------------------------------------

def state_prob(P0: float, rho: float, k: int, i: int) -> float:
    """
    Вероятность состояния P_i для СМО M/M/k/∞.

    При i ≤ k (заявки только обслуживаются):
        P_i = (ρ^i / i!) · P_0
    При i > k (часть заявок в очереди):
        P_i = ρ^i / (k! · k^(i-k)) · P_0
    """
    if i <= k:
        return (rho ** i) / math.factorial(i) * P0
    else:
        return (rho ** i) / (math.factorial(k) * k ** (i - k)) * P0


def mmk_characteristics(lam: float, mu: float, k: int) -> dict:
    """
    Полный набор характеристик СМО M/M/k/∞ при k каналах.

    Параметры
    ----------
    lam : λ — интенсивность входного потока, заявок/мин
    mu  : μ — интенсивность обслуживания одного канала, заявок/мин
    k   : число каналов

    Условие стационарности: χ = ρ/k < 1, где ρ = λ/μ.
    Если условие не выполнено — возвращается None.

    Используются формулы:
        ρ = λ/μ,   χ = ρ/k
        P_0 = 1 / [ Σ_{i=0..k-1} ρ^i/i!  +  ρ^k / (k!·(1-χ)) ]
        P_k = (ρ^k / k!) · P_0
        P_оч = P_k / (1-χ)               — формула Эрланга C, P(ожидания)
        L_оч = P_оч · χ / (1-χ)          — среднее число в очереди
        L_сист = L_оч + ρ                — среднее число в системе
        T_оч = L_оч / λ                  — среднее время ожидания (Литтл)
        T_сист = T_оч + 1/μ              — среднее время пребывания в системе
    """
    rho = lam / mu
    chi = rho / k                                # χ — загрузка одного канала
    if chi >= 1:
        return None                              # стационарного режима нет

    # P_0 — вероятность простоя всех каналов
    sum_finite = sum((rho ** i) / math.factorial(i) for i in range(k))
    tail_term = (rho ** k) / (math.factorial(k) * (1 - chi))
    P0 = 1.0 / (sum_finite + tail_term)

    # P_k — вероятность занятости всех k каналов (но без очереди)
    Pk = (rho ** k) / math.factorial(k) * P0

    # P_оч — вероятность того, что прибывшая заявка попадёт в очередь
    P_wait = Pk / (1 - chi)

    # L_оч — среднее число заявок в очереди
    L_q = P_wait * chi / (1 - chi)

    # L_сист — среднее число заявок в системе (в очереди + на обслуживании)
    L = L_q + rho

    # T_оч — среднее время ожидания (формула Литтла)
    T_q = L_q / lam

    # T_сист — среднее время пребывания заявки в системе
    T = T_q + 1.0 / mu

    return {
        "rho": rho, "chi": chi,
        "P0": P0, "Pk": Pk, "P_wait": P_wait,
        "L_q": L_q, "L": L,
        "T_q": T_q, "T": T,
    }


def find_k_min(rho: float) -> int:
    """
    Минимальное k, при котором система стационарна (ρ/k < 1).
    Т.е. наименьшее целое k > ρ.
    """
    return int(math.floor(rho)) + 1


def states_until_eps(P0: float, rho: float, k: int,
                     eps: float = EPS) -> list:
    """
    Вычисляет последовательно P_0, P_1, P_2, ... до тех пор,
    пока остаток ряда R_N = 1 − Σ_{i=0..N} P_i не станет меньше eps.

    Это реализация методического требования об оценке остатка
    бесконечного ряда: для i ≥ k вероятности образуют геометрическую
    прогрессию со знаменателем χ < 1, поэтому ряд сходится.
    """
    probs = []
    cumulative = 0.0
    i = 0
    while True:
        p = state_prob(P0, rho, k, i)
        probs.append(p)
        cumulative += p
        # Останавливаемся, когда хвост ряда мал и мы уже прошли k
        if i >= k and (1.0 - cumulative) < eps:
            break
        i += 1
        if i > 200:                              # защита от зацикливания
            break
    return probs


def prob_queue_leq_n(P0: float, rho: float, k: int, n: int) -> float:
    """
    Вероятность того, что в очереди находится не более n заявок.
    «В очереди ≤ n» эквивалентно «в системе ≤ k + n», т.к. первые k
    заявок обслуживаются на каналах.
        P(L_оч ≤ n) = Σ_{i=0..k+n} P_i
    """
    return sum(state_prob(P0, rho, k, i) for i in range(k + n + 1))


def cost(lam: float, mu: float, a: float, k: int) -> float:
    """
    Относительная стоимость C(k) = k/λ + a·T_оч.
    Если режим нестационарный, возвращает +∞.
    """
    r = mmk_characteristics(lam, mu, k)
    if r is None:
        return float("inf")
    return k / lam + a * r["T_q"]


def find_k_opt(lam: float, mu: float, a: float,
               k_start: int, k_end: int = 20) -> tuple:
    """
    Поиск k_opt в диапазоне [k_start, k_end] методом перебора.
    Возвращает (k_opt, C_opt, costs), где costs — список (k, C(k)).
    """
    costs = [(k, cost(lam, mu, a, k)) for k in range(k_start, k_end + 1)]
    k_opt, c_opt = min(costs, key=lambda kc: kc[1])
    return k_opt, c_opt, costs


# ----------------------------------------------------------------------
# Функции вывода (таблицы)
# ----------------------------------------------------------------------

def print_state_table(probs: list, eps: float) -> None:
    """Таблица предельных вероятностей состояний с контрольной суммой."""
    print(f"  {'Состояние':<11} | {'Вероятность':>13}")
    print(f"  {'-' * 11}-+-{'-' * 13}")
    for i, p in enumerate(probs):
        print(f"  P_{i:<9} | {fmt_mid(p):>13}")
    s = sum(probs)
    print(f"  {'Σ P_i':<11} | {fmt_mid(s):>13}")
    print(f"  Остаток R = 1 − Σ ≈ {fmt_mid(1 - s)}  (требование R < ε = {eps})")
    print(f"  Учтено {len(probs)} состояний (до выполнения критерия сходимости).")


def print_char_block(label: str, lam: float, mu: float, k: int,
                     eps: float = EPS) -> dict:
    """Полный отчёт по характеристикам СМО для заданного k."""
    print("\n" + "=" * 72)
    print(f"  {label}")
    print("=" * 72)
    r = mmk_characteristics(lam, mu, k)
    if r is None:
        print(f"  Режим не стационарный (ρ/k = {fmt_mid(lam / mu / k)} ≥ 1).")
        return r

    print(f"Число каналов:                            k = {k}")
    print(f"Приведённая интенсивность нагрузки:       ρ = {fmt_mid(r['rho'])}")
    print(f"Коэффициент загрузки канала:              χ = ρ/k = {fmt_mid(r['chi'])}")
    print(f"Вероятность простоя всех каналов:         P_0 = {fmt_mid(r['P0'])}")
    print(f"Вероятность занятости всех каналов:       P_k = {fmt_mid(r['Pk'])}")
    print(f"Вероятность ожидания в очереди:           P_оч = {fmt_mid(r['P_wait'])}")
    print()
    print("Предельные вероятности состояний:")
    probs = states_until_eps(r["P0"], r["rho"], k, eps)
    print_state_table(probs, eps)
    print()
    print("ОКОНЧАТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ (3 знака):")
    print(f"  Среднее число заявок в очереди       L_оч  = {fmt_fin(r['L_q'])}")
    print(f"  Среднее число заявок в системе       L_сист= {fmt_fin(r['L'])}")
    print(f"  Среднее время ожидания               T_оч  = {fmt_fin(r['T_q'])} мин")
    print(f"  Среднее время в системе              T_сист= {fmt_fin(r['T'])} мин")
    return r


# ----------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------

def plot_cost_and_chars(lam, mu, a, k_min, k_opt, costs, out_path):
    """График стоимости C(k) и основных характеристик от числа каналов."""
    ks = [kc[0] for kc in costs]
    cs = [kc[1] for kc in costs]
    L_q = [mmk_characteristics(lam, mu, k)["L_q"] for k in ks]
    T_q = [mmk_characteristics(lam, mu, k)["T_q"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Левая панель: функция стоимости ----
    ax = axes[0]
    ax.plot(ks, cs, "o-", color="steelblue", label="C(k)")
    ax.axvline(k_min, color="green", linestyle="--",
               label=f"k_min = {k_min}")
    ax.axvline(k_opt, color="red", linestyle=":", linewidth=2,
               label=f"k_opt = {k_opt}")
    ax.set_xlabel("Число каналов k")
    ax.set_ylabel("C(k) = k/λ + a·T_оч")
    ax.set_title("Относительная стоимость C(k)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- Правая панель: L_q и T_q ----
    ax = axes[1]
    ax.plot(ks, L_q, "s-", color="crimson", label="L_оч(k)")
    ax.set_xlabel("Число каналов k")
    ax.set_ylabel("Среднее число заявок в очереди L_оч",
                  color="crimson")
    ax.tick_params(axis="y", labelcolor="crimson")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(ks, T_q, "^-", color="darkorange", label="T_оч(k)")
    ax2.set_ylabel("Среднее время ожидания T_оч, мин",
                   color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax.axvline(k_min, color="green", linestyle="--", alpha=0.5)
    ax.axvline(k_opt, color="red", linestyle=":", linewidth=2, alpha=0.7)
    ax.set_title("Характеристики очереди")

    fig.suptitle(f"Зависимости показателей СМО M/M/k/∞ от k (ρ = "
                 f"{lam / mu:.4f})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_state_probs(P0, rho, k, eps, out_path):
    """Столбчатая диаграмма предельных вероятностей состояний."""
    probs = states_until_eps(P0, rho, k, eps)
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(probs)))
    bars = ax.bar(xs, probs, color="steelblue", edgecolor="navy")
    # На столбцах — числовые значения
    for i, (x, p) in enumerate(zip(xs, probs)):
        ax.text(x, p + max(probs) * 0.01, fmt_mid(p),
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Состояние i (число заявок в системе)")
    ax.set_ylabel("Предельная вероятность P_i")
    ax.set_title(f"Распределение предельных вероятностей "
                 f"состояний (k = {k}, ρ = {rho:.4f})")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Главная программа
# ----------------------------------------------------------------------

def main() -> None:
    # ---------- Исходные данные ----------
    lam_day = 88           # λ — заявок в сутки
    t_min = 2              # t — среднее время обслуживания, мин
    a = 6                  # a — коэффициент стоимости очереди
    n_queue = 4            # n — порог для P(в очереди ≤ n)

    # ---------- Приведение к единым единицам (минуты) ----------
    # 1 сутки = 24·60 = 1440 минут
    lam = lam_day / 1440.0     # λ — заявок/мин
    mu = 1.0 / t_min           # μ — заявок/мин
    rho = lam / mu             # приведённая интенсивность нагрузки

    print("ИСХОДНЫЕ ДАННЫЕ И РАСЧЁТНЫЕ ПАРАМЕТРЫ")
    print("-" * 72)
    print(f"  Интенсивность потока:                 λ = {lam_day} заявок/сутки")
    print(f"  Среднее время обслуживания:           t = {t_min} мин")
    print(f"  Коэффициент стоимости очереди:        a = {a}")
    print(f"  Порог числа заявок в очереди:         n = {n_queue}")
    print()
    print(f"  Приведённая к минутам:")
    print(f"     λ = {lam_day}/1440 = {fmt_mid(lam)} заявок/мин")
    print(f"     μ = 1/t = {fmt_mid(mu)} заявок/мин")
    print(f"     ρ = λ/μ = {fmt_mid(rho)}")

    # ---------- Задание 1. k_min и характеристики при k_min ----------
    k_min = find_k_min(rho)
    print("\n" + "=" * 72)
    print(f"  ЗАДАНИЕ 1. Минимальное число каналов и характеристики")
    print("=" * 72)
    print(f"Условие стационарности: ρ/k < 1, т.е. k > ρ = {fmt_mid(rho)}.")
    print(f"Минимальное целое k, удовлетворяющее условию: k_min = {k_min}")

    r_min = print_char_block(
        f"Характеристики СМО при k_min = {k_min}", lam, mu, k_min)

    # ---------- Задание 2. Поиск k_opt ----------
    k_opt, c_opt, costs = find_k_opt(lam, mu, a, k_min, k_end=15)

    print("\n" + "=" * 72)
    print(f"  ЗАДАНИЕ 2. Оптимальное число каналов")
    print("=" * 72)
    print(f"Функция стоимости: C(k) = k/λ + a·T_оч  (a = {a}, T_оч в мин)")
    print(f"\nТаблица 2. Зависимость стоимости C(k) от числа каналов k:")
    print(f"  {'k':>3} | {'L_оч':>10} | {'T_оч, мин':>12} | "
          f"{'k/λ':>12} | {'a·T_оч':>10} | {'C(k)':>12}")
    print(f"  {'-' * 3}-+-{'-' * 10}-+-{'-' * 12}-+-"
          f"{'-' * 12}-+-{'-' * 10}-+-{'-' * 12}")
    for k, c in costs:
        r = mmk_characteristics(lam, mu, k)
        term1 = k / lam
        term2 = a * r["T_q"]
        mark = "  ← min" if k == k_opt else ""
        print(f"  {k:>3} | {fmt_mid(r['L_q']):>10} | "
              f"{fmt_mid(r['T_q']):>12} | {fmt_mid(term1):>12} | "
              f"{fmt_mid(term2):>10} | {fmt_mid(c):>12}{mark}")

    print(f"\nОТВЕТ (зад. 2): k_opt = {k_opt}, "
          f"C(k_opt) = {fmt_fin(c_opt)}")

    # ---------- Задание 3. Сравнение k_min и k_opt ----------
    r_opt = mmk_characteristics(lam, mu, k_opt)
    print("\n" + "=" * 72)
    print(f"  ЗАДАНИЕ 3. Сравнение характеристик при k_min и k_opt")
    print("=" * 72)
    print(f"\n  {'Показатель':<35} | {'k_min = ' + str(k_min):>14} | "
          f"{'k_opt = ' + str(k_opt):>14}")
    print(f"  {'-' * 35}-+-{'-' * 14}-+-{'-' * 14}")
    rows = [
        ("Загрузка канала χ = ρ/k",           r_min["chi"],     r_opt["chi"]),
        ("P_0 (простой всех каналов)",        r_min["P0"],      r_opt["P0"]),
        ("P_оч (вероятность ожидания)",       r_min["P_wait"],  r_opt["P_wait"]),
        ("L_оч (среднее число в очереди)",    r_min["L_q"],     r_opt["L_q"]),
        ("L_сист (среднее число в системе)",  r_min["L"],       r_opt["L"]),
        ("T_оч (время ожидания, мин)",        r_min["T_q"],     r_opt["T_q"]),
        ("T_сист (время в системе, мин)",     r_min["T"],       r_opt["T"]),
        ("C(k) (относительная стоимость)",
            cost(lam, mu, a, k_min), cost(lam, mu, a, k_opt)),
    ]
    for name, v1, v2 in rows:
        print(f"  {name:<35} | {fmt_fin(v1):>14} | {fmt_fin(v2):>14}")

    # ---------- Задание 4. P(в очереди ≤ n) ----------
    print("\n" + "=" * 72)
    print(f"  ЗАДАНИЕ 4. Вероятность P(в очереди ≤ {n_queue} заявок)")
    print("=" * 72)
    # Считаем для k_opt (как для рекомендуемой конфигурации)
    # и параллельно для k_min — для полноты сравнения.
    for label, k_use in [(f"при k_min = {k_min}", k_min),
                         (f"при k_opt = {k_opt}", k_opt)]:
        r = mmk_characteristics(lam, mu, k_use)
        P_le_n = prob_queue_leq_n(r["P0"], r["rho"], k_use, n_queue)
        # Покажем слагаемые этой суммы
        print(f"\n  Расчёт {label}:")
        print(f"    P(L_оч ≤ {n_queue}) = Σ_{{i=0..{k_use + n_queue}}} P_i")
        contrib = []
        for i in range(k_use + n_queue + 1):
            pi = state_prob(r["P0"], r["rho"], k_use, i)
            contrib.append(pi)
            print(f"      P_{i} = {fmt_mid(pi)}")
        print(f"    Σ = {fmt_mid(sum(contrib))}")
        print(f"    ОТВЕТ: P(в очереди ≤ {n_queue}) = {fmt_fin(P_le_n)}")

    # ---------- Графики ----------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    g1 = os.path.join(out_dir, "task2_cost_and_chars.png")
    g2 = os.path.join(out_dir, "task2_state_probs.png")
    plot_cost_and_chars(lam, mu, a, k_min, k_opt, costs, g1)
    plot_state_probs(r_min["P0"], r_min["rho"], k_min, EPS, g2)
    print(f"\nГрафики сохранены:")
    print(f"  • {g1}")
    print(f"  • {g2}")

    # ---------- Выводы ----------
    print("\n" + "=" * 72)
    print("  ВЫВОДЫ")
    print("=" * 72)
    print(
        f"1. Приведённая интенсивность нагрузки ρ = {fmt_mid(rho)} весьма мала "
        f"(≈ 12 % от пропускной способности одного канала). Поэтому "
        f"минимальное число каналов для стационарного режима очень скромное: "
        f"k_min = {k_min}."
    )
    print(
        f"2. При k_min = {k_min} коэффициент загрузки канала χ = "
        f"{fmt_fin(r_min['chi'])}, вероятность ожидания в очереди "
        f"P_оч = {fmt_fin(r_min['P_wait'])} ({fmt_fin(r_min['P_wait'] * 100)} %). "
        f"Среднее число заявок в очереди L_оч = {fmt_fin(r_min['L_q'])}, "
        f"среднее время ожидания T_оч = {fmt_fin(r_min['T_q'])} мин — это "
        f"меньше {fmt_fin(r_min['T_q'] * 60)} секунд."
    )
    print(
        f"3. Оптимизация по стоимости C(k) = k/λ + a·T_оч дала k_opt = "
        f"{k_opt}, что совпадает с k_min. Причина: при малой нагрузке "
        f"добавление каналов лишь увеличивает первое слагаемое k/λ, тогда "
        f"как T_оч и так пренебрежимо мало. Любые дополнительные каналы "
        f"были бы экономически не оправданы."
    )
    print(
        f"4. P(в очереди ≤ {n_queue}) при k = {k_min} равна "
        f"{fmt_fin(prob_queue_leq_n(r_min['P0'], r_min['rho'], k_min, n_queue))} — "
        f"то есть событие практически достоверное; реально в очереди почти "
        f"никогда не оказывается больше {n_queue} заявок."
    )
    print(
        f"5. Влияние параметров: при росте λ или t (увеличение ρ) растут и "
        f"L_оч, и T_оч, что увеличивает второе слагаемое в C(k); тогда "
        f"оптимальное k растёт. При уменьшении t (рост μ) оптимальное k "
        f"снижается. Коэффициент a задаёт компромисс между стоимостью "
        f"каналов и стоимостью ожидания клиентов: чем больше a, тем "
        f"выгоднее увеличивать k."
    )


if __name__ == "__main__":
    main()
