"""
Лабораторная работа по теории массового обслуживания.

Задача 6. Замкнутая многоканальная СМО.

Расчёт по формулам учебного пособия
М. А. Плескунова «Теория массового обслуживания» (2022).

МОДЕЛЬ:
    n источников заявок и k каналов обслуживания.
    Каждый активный источник создаёт заявку с интенсивностью λ.
    Время обслуживания заявки — экспоненциальное со средним 1/μ = t.

    Процесс гибели и размножения с переменными интенсивностями
    (состояние i — число заявок в системе):
        λ_i = (n - i) · λ           (i = 0..n)
        μ_i = min(i, k) · μ         (i = 1..n)

    Стационарные вероятности:
        для i ≤ k:   P_i = P_0 · C(n, i) · ρ^i,            ρ = λ/μ
        для i > k:   P_i = P_0 · n! / [(n-i)! · k! · k^(i-k)] · ρ^i
    P_0 определяется из условия нормировки Σ P_i = 1.

ИСХОДНЫЕ ДАННЫЕ:
    k = 5     — число каналов
    n = 24    — число источников заявок
    λ = 1.5   — интенсивность поступления заявок от одного активного
                источника, заявок/час
    t = 0.5   — среднее время обслуживания, час

Точность вывода:
    • промежуточные вычисления — 5 знаков;
    • окончательные ответы     — 3 знака.
"""

import math
import os
import matplotlib.pyplot as plt


PREC_MID = 5
PREC_FIN = 3


def fmt_mid(x): return f"{x:.{PREC_MID}f}"
def fmt_fin(x): return f"{x:.{PREC_FIN}f}"
def fmt_sci(x):
    if x == 0 or x >= 1e-4:
        return fmt_mid(x)
    return f"{x:.3e}"


# ----------------------------------------------------------------------
# Расчётные функции
# ----------------------------------------------------------------------

def closed_multichannel_probabilities(n: int, k: int,
                                      lam: float, mu: float) -> list:
    """
    Стационарные вероятности замкнутой k-канальной СМО с n источниками.

    Используется рекуррентное соотношение, полученное из уравнений
    детального равновесия:
        для i = 1..k:    q_i = q_{i-1} · (n - i + 1) · ρ / i
        для i = k+1..n:  q_i = q_{i-1} · (n - i + 1) · ρ / k
    где q_i — ненормированные q_i (q_0 = 1). После расчёта всех q_i
    производится нормировка: P_i = q_i / Σ q_i.

    Возвращает список [P_0, P_1, ..., P_n].
    """
    rho = lam / mu
    q = [1.0]                                   # q_0 = 1
    for i in range(1, n + 1):
        if i <= k:
            q.append(q[-1] * (n - i + 1) * rho / i)
        else:
            q.append(q[-1] * (n - i + 1) * rho / k)
    Z = sum(q)
    return [qi / Z for qi in q]


def closed_multichannel_characteristics(n: int, k: int,
                                         lam: float, mu: float) -> dict:
    """
    Полный набор характеристик замкнутой k-канальной СМО.

    Характеристики:
        P              — список предельных вероятностей P_0..P_n;
        P_free         — вероятность, что все каналы свободны = P_0;
        L_оч           — среднее число заявок в очереди
                          = Σ_{i>k} (i-k)·P_i;
        L_сист         — среднее число заявок в системе = Σ i·P_i;
        L_занят        — среднее число занятых каналов
                          = Σ min(i,k)·P_i  (= A/μ);
        L_свободн      — среднее число свободных каналов = k - L_занят;
        A              — абсолютная пропускная способность = μ·L_занят
                          (контроль: A = λ·(n - L_сист));
        Q              — относительная пропускная способность
                          = A / (n·λ)  (доля потенциального спроса);
        P_оч           — вероятность наличия очереди = P(i > k);
        T_оч           — среднее время ожидания = L_оч / A;
        T_обс          — среднее время обслуживания = 1/μ;
        T_сист         — среднее время в системе = L_сист / A
                          (= T_оч + T_обс).
    """
    rho = lam / mu
    P = closed_multichannel_probabilities(n, k, lam, mu)

    P_free = P[0]                                # все каналы свободны (i = 0)

    # Средние количества по состояниям
    L_q = sum((i - k) * P[i] for i in range(k + 1, n + 1))
    L_sys = sum(i * P[i] for i in range(n + 1))
    L_busy = sum(min(i, k) * P[i] for i in range(n + 1))
    L_free = k - L_busy

    # Пропускные способности
    A = mu * L_busy                               # абсолютная (заявок/час)
    A_check = lam * (n - L_sys)                   # контроль через формулу Литтла
    Q = A / (n * lam)                             # относительная (доля)
    Q_chan = L_busy / k                           # загрузка каналов (для справки)

    # Вероятность наличия очереди
    P_queue = sum(P[i] for i in range(k + 1, n + 1))

    # Времена (формула Литтла; λ_эфф = A — все заявки в замкнутой системе
    # обслуживаются, поэтому эффективный поток равен пропускной способности)
    T_serv = 1.0 / mu
    T_q = L_q / A if A > 0 else 0.0
    T_sys = L_sys / A if A > 0 else 0.0

    return {
        "rho": rho, "P": P,
        "P_free": P_free, "P_queue": P_queue,
        "L_q": L_q, "L_sys": L_sys, "L_busy": L_busy, "L_free": L_free,
        "A": A, "A_check": A_check, "Q": Q, "Q_chan": Q_chan,
        "T_q": T_q, "T_serv": T_serv, "T_sys": T_sys,
    }


# ----------------------------------------------------------------------
# Функции вывода
# ----------------------------------------------------------------------

def print_state_table(P: list, k: int, n: int) -> None:
    """Таблица всех (n+1) состояний с указанием — что они означают."""
    print(f"  {'i':<4} | {'P_i':>13} | Описание состояния")
    print(f"  {'-' * 4}-+-{'-' * 13}-+-{'-' * 40}")
    for i, p in enumerate(P):
        if i == 0:
            descr = "все каналы свободны"
        elif i < k:
            descr = f"занято {i} из {k} каналов, очереди нет"
        elif i == k:
            descr = f"все {k} каналов заняты, очереди нет"
        else:
            descr = f"в очереди {i - k} заявок(а)"
        print(f"  {i:<4} | {fmt_sci(p):>13} | {descr}")
    s = sum(P)
    print(f"  {'Σ':<4} | {fmt_mid(s):>13} | контроль (должна быть 1)")


# ----------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------

def plot_state_distribution(P: list, k: int, n: int, out_path: str) -> None:
    """Распределение вероятностей состояний."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    xs = list(range(len(P)))
    # Серый — пустая система, синий — каналы, оранжевый — все каналы,
    # красный — очередь
    colors = []
    for i in xs:
        if i == 0:
            colors.append("lightgray")
        elif i < k:
            colors.append("steelblue")
        elif i == k:
            colors.append("orange")
        else:
            colors.append("crimson")
    ax.bar(xs, P, color=colors, edgecolor="navy", alpha=0.85)
    for i, p in enumerate(P):
        if p > max(P) * 0.02:
            ax.text(i, p + max(P) * 0.01, fmt_fin(p),
                    ha="center", va="bottom", fontsize=8)
    ax.axvline(k - 0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(k + 0.5, color="orange", linestyle="--", alpha=0.6,
               label="граница появления очереди")
    ax.set_xlabel("i — число заявок в системе")
    ax.set_ylabel("Предельная вероятность P_i")
    ax.set_xticks(xs)
    ax.set_title(f"Распределение вероятностей состояний замкнутой "
                 f"k-канальной СМО (k = {k}, n = {n})\n"
                 f"Серый — пусто; синий — занятые каналы; "
                 f"оранжевый — все каналы заняты; красный — очередь")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_to_k(n, lam, mu, k_orig, out_path):
    """График зависимости основных показателей от числа каналов k."""
    ks = list(range(1, min(n, 15) + 1))
    L_q_vals, L_sys_vals, A_vals, Q_vals, Pq_vals, T_q_vals = (
        [], [], [], [], [], []
    )
    for k in ks:
        r = closed_multichannel_characteristics(n, k, lam, mu)
        L_q_vals.append(r["L_q"])
        L_sys_vals.append(r["L_sys"])
        A_vals.append(r["A"])
        Q_vals.append(r["Q"])
        Pq_vals.append(r["P_queue"])
        T_q_vals.append(r["T_q"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Левая панель: L_оч, L_сист, T_оч
    ax = axes[0]
    ax.plot(ks, L_q_vals, "o-", color="crimson", label="L_оч(k)")
    ax.plot(ks, L_sys_vals, "s-", color="steelblue", label="L_сист(k)")
    ax.axvline(k_orig, color="orange", linestyle=":",
               label=f"k = {k_orig} (задано)")
    ax.set_xlabel("Число каналов k")
    ax.set_ylabel("Средние количества заявок")
    ax.set_title("Среднее число заявок в очереди и в системе")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Правая панель: A, Q, P_оч
    ax = axes[1]
    ax.plot(ks, A_vals, "o-", color="seagreen", label="A(k), заявок/час")
    ax.axhline(n * lam, color="gray", linestyle="--", alpha=0.5,
               label=f"потенциал n·λ = {n * lam}")
    ax.set_xlabel("Число каналов k")
    ax.set_ylabel("A, заявок/час", color="seagreen")
    ax.tick_params(axis="y", labelcolor="seagreen")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ks, Q_vals, "s-", color="crimson", label="Q(k)")
    ax2.plot(ks, Pq_vals, "^-", color="purple", label="P_оч(k)")
    ax2.set_ylabel("Q и P_оч (доля)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim(0, 1.05)
    ax.axvline(k_orig, color="orange", linestyle=":", alpha=0.7)
    ax.set_title("Пропускная способность и вероятность очереди")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right",
              fontsize=9)

    fig.suptitle(f"Зависимости показателей от числа каналов k "
                 f"(n = {n}, λ = {lam}, μ = {mu})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Главная программа
# ----------------------------------------------------------------------

def main() -> None:
    # ---------- Исходные данные ----------
    k = 5
    n = 24
    lam = 1.5            # заявок/час на один активный источник
    t = 0.5              # часов на одну заявку

    # ---------- Производные параметры ----------
    mu = 1.0 / t         # заявок/час, на один канал
    rho = lam / mu

    print("ИСХОДНЫЕ ДАННЫЕ И РАСЧЁТНЫЕ ПАРАМЕТРЫ")
    print("-" * 72)
    print(f"  Число каналов:                        k = {k}")
    print(f"  Число источников заявок:              n = {n}")
    print(f"  Интенсивность одного источника:       λ = {lam} заявок/час")
    print(f"  Среднее время обслуживания:           t = {t} час")
    print()
    print(f"  Производные параметры:")
    print(f"     μ = 1/t   = {fmt_mid(mu)} заявок/час")
    print(f"     ρ = λ/μ   = {fmt_mid(rho)}")
    print(f"     Потенциал спроса n·λ = {fmt_mid(n * lam)} заявок/час")
    print(f"     Мощность СМО k·μ      = {fmt_mid(k * mu)} заявок/час")
    print(f"     Отношение n·λ / k·μ   = {fmt_mid(n * lam / (k * mu))}  "
          f"(> 1 ⇒ очередь существенна)")

    # ---------- Расчёт ----------
    r = closed_multichannel_characteristics(n, k, lam, mu)

    # ---------- Пункт 1. Предельные вероятности ----------
    print("\n" + "=" * 72)
    print("  ПУНКТ 1. Предельные вероятности состояний СМО")
    print("=" * 72)
    print("Таблица 1. Распределение P_i:")
    print_state_table(r["P"], k, n)

    # ---------- Пункт 2. Все каналы свободны ----------
    print("\n" + "=" * 72)
    print("  ПУНКТ 2. Вероятность того, что все каналы свободны")
    print("=" * 72)
    print(f"  Это вероятность состояния «в системе нет заявок» (i = 0):")
    print(f"  ОТВЕТ: P(все каналы свободны) = P_0 = "
          f"{fmt_fin(r['P_free'])}    "
          f"({fmt_sci(r['P_free'])})")

    # ---------- Пункт 3. Среднее число заявок в очереди ----------
    print("\n" + "=" * 72)
    print("  ПУНКТ 3. Среднее число заявок в очереди")
    print("=" * 72)
    print(f"  L_оч = Σ_{{i={k + 1}..{n}}} (i − k)·P_i")
    print(f"  ОТВЕТ: L_оч = {fmt_fin(r['L_q'])}")

    # ---------- Пункт 4. Среднее число заявок в системе ----------
    print("\n" + "=" * 72)
    print("  ПУНКТ 4. Среднее число заявок в системе")
    print("=" * 72)
    print(f"  L_сист = Σ_{{i=0..{n}}} i·P_i")
    print(f"  ОТВЕТ: L_сист = {fmt_fin(r['L_sys'])}")

    # ---------- Пункт 5. Среднее число свободных каналов ----------
    print("\n" + "=" * 72)
    print("  ПУНКТ 5. Среднее число свободных каналов")
    print("=" * 72)
    print(f"  L_своб = k − L_занят")
    print(f"  ОТВЕТ: L_своб = {fmt_fin(r['L_free'])}")

    # ---------- Пункт 6. Среднее число занятых каналов ----------
    print("\n" + "=" * 72)
    print("  ПУНКТ 6. Среднее число занятых каналов")
    print("=" * 72)
    print(f"  L_занят = Σ min(i, k)·P_i")
    print(f"  ОТВЕТ: L_занят = {fmt_fin(r['L_busy'])}    "
          f"(загрузка каналов: {fmt_fin(r['Q_chan'] * 100)} %)")

    # ---------- Пункт 7. Абсолютная пропускная способность ----------
    print("\n" + "=" * 72)
    print("  ПУНКТ 7. Абсолютная пропускная способность")
    print("=" * 72)
    print(f"  A = μ·L_занят")
    print(f"  Контроль: A = λ·(n − L_сист) = "
          f"{fmt_mid(r['A_check'])} (совпадает)")
    print(f"  ОТВЕТ: A = {fmt_fin(r['A'])} заявок/час")

    # ---------- Пункт 8. Относительная пропускная способность ----------
    print("\n" + "=" * 72)
    print("  ПУНКТ 8. Относительная пропускная способность")
    print("=" * 72)
    print(f"  Q = A / (n·λ)  (доля удовлетворяемого «потенциального» спроса)")
    print(f"  Расчёт: Q = {fmt_mid(r['A'])} / {n * lam} = "
          f"{fmt_mid(r['Q'])}")
    print(f"  ОТВЕТ: Q = {fmt_fin(r['Q'])}    "
          f"({fmt_fin(r['Q'] * 100)} %)")

    # ---------- Пункт 9. Вероятность наличия очереди ----------
    print("\n" + "=" * 72)
    print("  ПУНКТ 9. Вероятность наличия очереди")
    print("=" * 72)
    print(f"  P_оч = P(i > k) = Σ_{{i={k + 1}..{n}}} P_i")
    print(f"  ОТВЕТ: P_оч = {fmt_fin(r['P_queue'])}    "
          f"({fmt_fin(r['P_queue'] * 100)} %)")

    # ---------- Пункт 10. Времена ----------
    print("\n" + "=" * 72)
    print("  ПУНКТ 10. Среднее время ожидания, обслуживания, в системе")
    print("=" * 72)
    print(f"  Формула Литтла с эффективным потоком λ_эфф = A:")
    print(f"     T_оч  = L_оч  / A = {fmt_mid(r['L_q'])} / "
          f"{fmt_mid(r['A'])} = {fmt_mid(r['T_q'])} час")
    print(f"     T_обс = 1/μ                                  "
          f"= {fmt_mid(r['T_serv'])} час")
    print(f"     T_сис = L_сист / A = {fmt_mid(r['L_sys'])} / "
          f"{fmt_mid(r['A'])} = {fmt_mid(r['T_sys'])} час")
    print(f"     Контроль:  T_оч + T_обс = "
          f"{fmt_mid(r['T_q'] + r['T_serv'])} (= T_сис ✓)")
    print()
    print(f"  ОТВЕТЫ (3 знака):")
    print(f"     T_оч  = {fmt_fin(r['T_q'])} час "
          f"= {fmt_fin(r['T_q'] * 60)} мин")
    print(f"     T_обс = {fmt_fin(r['T_serv'])} час "
          f"= {fmt_fin(r['T_serv'] * 60)} мин")
    print(f"     T_сис = {fmt_fin(r['T_sys'])} час "
          f"= {fmt_fin(r['T_sys'] * 60)} мин")

    # ---------- Графики ----------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    g1 = os.path.join(out_dir, "task6_state_distribution.png")
    g2 = os.path.join(out_dir, "task6_sensitivity_k.png")
    plot_state_distribution(r["P"], k, n, g1)
    plot_sensitivity_to_k(n, lam, mu, k, g2)
    print(f"\nГрафики сохранены:")
    print(f"  • {g1}")
    print(f"  • {g2}")

    # ---------- Сводная таблица всех ответов ----------
    print("\n" + "=" * 72)
    print("  СВОДКА ОТВЕТОВ")
    print("=" * 72)
    rows = [
        ("Вероятность, что все каналы свободны  P_0",      r["P_free"]),
        ("Среднее число заявок в очереди        L_оч",     r["L_q"]),
        ("Среднее число заявок в системе        L_сист",   r["L_sys"]),
        ("Среднее число свободных каналов       L_своб",   r["L_free"]),
        ("Среднее число занятых каналов         L_занят",  r["L_busy"]),
        ("Абсолютная пропускная способность     A",        r["A"]),
        ("Относительная пропускная способность  Q",        r["Q"]),
        ("Вероятность наличия очереди           P_оч",     r["P_queue"]),
        ("Среднее время ожидания, час            T_оч",     r["T_q"]),
        ("Среднее время обслуживания, час        T_обс",    r["T_serv"]),
        ("Среднее время в системе, час           T_сис",    r["T_sys"]),
    ]
    for name, val in rows:
        print(f"  {name:<42}  =  {fmt_fin(val)}")

    # ---------- Выводы ----------
    print("\n" + "=" * 72)
    print("  ВЫВОДЫ")
    print("=" * 72)
    print(
        f"1. Система значительно перегружена «изнутри»: потенциал спроса "
        f"n·λ = {n * lam} заявок/час сильно превышает мощность каналов "
        f"k·μ = {k * mu} заявок/час (в {fmt_fin(n * lam / (k * mu))} раз). "
        f"Поэтому большинство источников оказывается в системе, а очередь "
        f"практически всегда не пуста: P_оч = {fmt_fin(r['P_queue'])} "
        f"({fmt_fin(r['P_queue'] * 100)} %)."
    )
    print(
        f"2. В среднем в системе L_сист = {fmt_fin(r['L_sys'])} заявок "
        f"(из {n} источников), причём L_занят = {fmt_fin(r['L_busy'])} "
        f"находится на каналах (загрузка "
        f"{fmt_fin(r['Q_chan'] * 100)} %), и L_оч = {fmt_fin(r['L_q'])} "
        f"в очереди. Свободных каналов в среднем "
        f"L_своб = {fmt_fin(r['L_free'])}."
    )
    print(
        f"3. Абсолютная пропускная способность A = {fmt_fin(r['A'])} "
        f"заявок/час — близка к предельной k·μ = {k * mu} (каналы "
        f"работают почти без простоя). Относительная пропускная "
        f"способность Q = {fmt_fin(r['Q'])} ({fmt_fin(r['Q'] * 100)} %) "
        f"означает: только эта доля «потенциала» n·λ реально "
        f"обслуживается; остальные источники ждут в системе."
    )
    print(
        f"4. Среднее время ожидания T_оч = {fmt_fin(r['T_q'] * 60)} мин "
        f"в {fmt_fin(r['T_q'] / r['T_serv'])} раз больше времени самого "
        f"обслуживания t = {fmt_fin(r['T_serv'] * 60)} мин. Полное время "
        f"в системе T_сис = {fmt_fin(r['T_sys'] * 60)} мин."
    )
    print(
        f"5. Влияние параметров (см. task6_sensitivity_k.png): при росте k "
        f"очередь сокращается, P_оч и L_оч резко падают, Q приближается "
        f"к 1. При снижении t (или росте μ) — аналогичный эффект. "
        f"Главный «рычаг» в данной системе — расширение числа каналов: "
        f"даже одно дополнительное место заметно меняет картину. "
        f"При уменьшении λ или n также наблюдается разгрузка."
    )


if __name__ == "__main__":
    main()
