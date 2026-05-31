"""
Лабораторная работа по теории массового обслуживания.

Задача 5. Замкнутая одноканальная СМО (модель «ремонтника»,
         частный случай модели Энгсета при k = 1 канале).

Расчёт по формулам учебного пособия
М. А. Плескунова «Теория массового обслуживания» (2022).

МОДЕЛЬ:
    n источников заявок и 1 канал обслуживания.
    Источник либо работает (исправен), либо находится в системе
    (на ремонте или в очереди).
    Каждый исправный источник генерирует заявки с интенсивностью λ.
    Время обслуживания заявки — экспоненциальное со средним 1/μ = t.

    Процесс гибели и размножения (i — число НЕисправных источников):
        интенсивность поломок:    λ_i = (n - i)·λ,    i = 0..n
        интенсивность ремонтов:   μ_i = μ,            i = 1..n

    Стационарное распределение:
        P_i = P_0 · n!/(n-i)! · ρ^i,    ρ = λ/μ
        P_0 = 1 / Σ_{i=0..n} n!/(n-i)! · ρ^i

ИСХОДНЫЕ ДАННЫЕ:
    n = 18    — число источников заявок
    k = 4     — среднее число заявок (поломок) в месяц (СУММАРНО по всем
                источникам); отсюда интенсивность поломки одного активного
                источника λ = k/n заявок/мес
    t = 1.5   — среднее время ремонта одной заявки, дней
    P = 80    — порог по доле активных источников, %

ТРЕБУЕТСЯ:
    1) Вероятность того, что не менее P процентов источников активны.
    2) Дополнительные характеристики:
       - среднее число неисправных источников L_неис (в ремонте + в очереди);
       - абсолютная пропускная способность A (среднее число обслуженных
         заявок в единицу времени);
       - среднее время ремонта T_рем и ожидания ремонта T_ож.

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

def closed_system_probabilities(n: int, lam: float, mu: float) -> list:
    """
    Стационарные вероятности замкнутой одноканальной СМО.

    Формулы:
        ρ = λ/μ
        P_i = P_0 · n!/(n-i)! · ρ^i,   i = 0..n
        P_0 — из условия нормировки Σ P_i = 1.

    Возвращает список [P_0, P_1, ..., P_n].
    """
    rho = lam / mu
    # Рекуррентное вычисление ненормированных q_i = n!/(n-i)! · ρ^i:
    #   q_0 = 1,  q_{i+1} = q_i · (n-i) · ρ
    q = [1.0]
    for i in range(1, n + 1):
        q.append(q[-1] * (n - i + 1) * rho)
    Z = sum(q)
    return [qi / Z for qi in q]


def closed_system_characteristics(n: int, lam: float, mu: float) -> dict:
    """
    Полный набор характеристик замкнутой одноканальной СМО.

    Характеристики:
        L_неис = Σ i · P_i              — среднее число неисправных
        L_оч   = Σ (i-1) · P_i (i≥1) = L_неис − (1 − P_0)
        L_рем  = 1 − P_0                 — среднее число на ремонте (≤1)
        A      = μ · (1 − P_0)           — абсолютная пропускная способность
        T_рем  = 1/μ                     — среднее время ремонта
        T_ож   = L_оч / A                — среднее время ожидания (Литтл)
        T_сис  = T_ож + T_рем            — общее время «в неисправности»
    """
    rho = lam / mu
    P = closed_system_probabilities(n, lam, mu)

    L_neis = sum(i * P[i] for i in range(n + 1))        # неисправные
    L_repair = 1.0 - P[0]                                # на ремонте
    L_queue = L_neis - L_repair                          # в очереди

    A = mu * (1.0 - P[0])                                # абс. пропускная способность
    T_repair = 1.0 / mu                                   # среднее время ремонта
    T_wait = L_queue / A if A > 0 else 0.0               # среднее время ожидания
    T_sys = T_wait + T_repair                             # среднее время в системе

    # Контроль баланса потоков: A должна равняться λ · (n − L_неис)
    A_check = lam * (n - L_neis)

    return {
        "rho": rho, "P": P,
        "L_neis": L_neis, "L_queue": L_queue, "L_repair": L_repair,
        "A": A, "A_check": A_check,
        "T_repair": T_repair, "T_wait": T_wait, "T_sys": T_sys,
    }


def prob_active_at_least(P: list, n: int, P_percent: float) -> tuple:
    """
    Вычисляет P(доля активных источников ≥ P_percent %).

    Условие: (n - i)/n ≥ P_percent/100  ⇔  i ≤ n·(1 - P_percent/100)
    Если правая часть нецелое, берём её floor (так как i — целое).

    Возвращает (вероятность, i_max — максимальное допустимое число
    неисправных источников).
    """
    threshold = n * (1.0 - P_percent / 100.0)
    i_max = math.floor(threshold)
    # Если порог точно целочисленный — берём его, так как
    # «не менее P%» включает значение P% (равенство допустимо)
    if abs(threshold - round(threshold)) < 1e-12:
        i_max = int(round(threshold))
    prob = sum(P[:i_max + 1])
    return prob, i_max


# ----------------------------------------------------------------------
# Функции вывода
# ----------------------------------------------------------------------

def print_state_table(P: list, n: int) -> None:
    """Таблица всех (n+1) состояний — много, но компактно."""
    print(f"  {'i (неисп.)':<12} | {'P_i':>13} | "
          f"{'Активных':>10} | {'Доля акт.':>10}")
    print(f"  {'-' * 12}-+-{'-' * 13}-+-{'-' * 10}-+-{'-' * 10}")
    for i, p in enumerate(P):
        active = n - i
        share = active / n
        print(f"  {i:<12} | {fmt_sci(p):>13} | "
              f"{active:>10} | {fmt_fin(share):>10}")
    print(f"  {'Σ P_i':<12} | {fmt_mid(sum(P)):>13} |"
          f"{' (контроль)':>22}")


# ----------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------

def plot_state_distribution(P: list, n: int, i_max: int,
                             P_percent: float, out_path: str) -> None:
    """Распределение вероятностей и выделение зоны «активных ≥ P%»."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    xs = list(range(n + 1))
    # Зелёный — состояния, удовлетворяющие порогу; серый — нет
    colors = ["seagreen" if i <= i_max else "lightgray" for i in xs]
    ax.bar(xs, P, color=colors, edgecolor="navy", alpha=0.85)
    for i, p in enumerate(P):
        if p > max(P) * 0.005:
            label = fmt_mid(p) if p >= 1e-4 else f"{p:.1e}"
            ax.text(i, p + max(P) * 0.01, label,
                    ha="center", va="bottom", fontsize=8)
    ax.axvline(i_max + 0.5, color="red", linestyle="--",
               label=f"граница i ≤ {i_max} (активных ≥ {P_percent}%)")
    ax.set_xlabel("i — число неисправных источников")
    ax.set_ylabel("Предельная вероятность P_i")
    ax.set_title(f"Распределение вероятностей состояний замкнутой "
                 f"одноканальной СМО (n = {n})\n"
                 f"Зелёным выделены состояния с долей активных ≥ "
                 f"{P_percent}%")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_prob_vs_threshold(P: list, n: int, P_percent_orig: float,
                            out_path: str) -> None:
    """График: как меняется P(активных ≥ p%) от порога p."""
    thresholds = list(range(0, 101))
    probs = []
    for p in thresholds:
        prob, _ = prob_active_at_least(P, n, p)
        probs.append(prob)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(thresholds, probs, "-", color="steelblue", linewidth=2)
    # Отметим заданное P
    prob_at_P, _ = prob_active_at_least(P, n, P_percent_orig)
    ax.axvline(P_percent_orig, color="red", linestyle="--", alpha=0.7,
               label=f"P = {P_percent_orig}%")
    ax.scatter([P_percent_orig], [prob_at_P], color="red", s=80, zorder=5)
    ax.annotate(f"  {fmt_fin(prob_at_P)}",
                (P_percent_orig, prob_at_P),
                fontsize=10, color="red", va="center")
    ax.set_xlabel("Порог доли активных источников p, %")
    ax.set_ylabel("P(доля активных ≥ p)")
    ax.set_title("Зависимость вероятности от порога доли активных")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Главная программа
# ----------------------------------------------------------------------

def main() -> None:
    # ---------- Исходные данные ----------
    n = 18              # число источников
    k_total = 4         # среднее число заявок в месяц (суммарно)
    t_days = 1.5        # среднее время ремонта, дней
    P_percent = 80      # требуемая доля активных источников, %

    # ---------- Производные параметры (в месяцах) ----------
    # 1 месяц = 30 дней
    mu = 30.0 / t_days                      # μ — заявок/мес (1/время ремонта)
    lam = k_total / n                        # λ — заявок/мес на ОДИН источник
    rho = lam / mu

    print("ИСХОДНЫЕ ДАННЫЕ И РАСЧЁТНЫЕ ПАРАМЕТРЫ")
    print("-" * 72)
    print(f"  Число источников:                     n = {n}")
    print(f"  Среднее число заявок в месяц (всего): k = {k_total}")
    print(f"  Среднее время ремонта:                t = {t_days} дней")
    print(f"  Требуемая доля активных:              P = {P_percent} %")
    print()
    print(f"  Производные параметры (единицы — месяцы):")
    print(f"     μ = 30/t  = {fmt_mid(mu)} заявок/мес  (интенсивность ремонта)")
    print(f"     λ = k/n   = {fmt_mid(lam)} заявок/мес  "
          f"(интенсивность поломки одного источника)")
    print(f"     ρ = λ/μ   = {fmt_mid(rho)}")
    print(f"     n·ρ       = {fmt_mid(n * rho)}  (нагрузка системы в "
          f"открытой аналогии)")

    # ---------- Расчёт распределения и характеристик ----------
    r = closed_system_characteristics(n, lam, mu)

    print("\n" + "=" * 72)
    print(f"  Предельные вероятности состояний")
    print("=" * 72)
    print("Таблица 1. Распределение вероятностей P_i по состояниям "
          f"(i = 0..{n}):")
    print_state_table(r["P"], n)

    # ---------- Задание 1. Вероятность «не менее P% активны» ----------
    prob, i_max = prob_active_at_least(r["P"], n, P_percent)

    print("\n" + "=" * 72)
    print(f"  ЗАДАНИЕ 1. P(активных ≥ {P_percent}% от n = {n})")
    print("=" * 72)
    threshold = n * (1 - P_percent / 100)
    print(f"Условие «активных ≥ {P_percent}%»:")
    print(f"     (n − i)/n ≥ {P_percent}/100  ⇒  "
          f"i ≤ n·(1 − P/100) = {fmt_mid(threshold)}")
    print(f"     Поскольку i — целое, берём i ≤ {i_max}.")
    print(f"\n  P(активных ≥ {P_percent}%) = Σ_{{i=0..{i_max}}} P_i")
    contrib = []
    for i in range(i_max + 1):
        contrib.append(r["P"][i])
        print(f"      P_{i} = {fmt_mid(r['P'][i])}")
    print(f"      Σ   = {fmt_mid(sum(contrib))}")
    print(f"\n  ОТВЕТ: P(активных ≥ {P_percent}%) = {fmt_fin(prob)}")

    # ---------- Задание 2. Дополнительные характеристики ----------
    print("\n" + "=" * 72)
    print(f"  ЗАДАНИЕ 2. Дополнительные характеристики")
    print("=" * 72)
    print(f"\nВ ЕДИНИЦАХ «МЕСЯЦ»:")
    print(f"  Среднее число неисправных источников   L_неис = "
          f"{fmt_fin(r['L_neis'])}")
    print(f"      в том числе на ремонте              L_рем  = "
          f"{fmt_fin(r['L_repair'])}")
    print(f"      в том числе в очереди               L_оч   = "
          f"{fmt_fin(r['L_queue'])}")
    print(f"  Абсолютная пропускная способность      A      = "
          f"{fmt_fin(r['A'])} заявок/мес")
    print(f"      контроль: A = λ·(n − L_неис) =         "
          f"{fmt_fin(r['A_check'])} заявок/мес (совпадает)")
    print(f"  Среднее время ремонта                  T_рем  = "
          f"{fmt_fin(r['T_repair'])} мес")
    print(f"  Среднее время ожидания ремонта         T_ож   = "
          f"{fmt_fin(r['T_wait'])} мес")
    print(f"  Среднее время в неисправности (T_ож+T_рем) T_сис = "
          f"{fmt_fin(r['T_sys'])} мес")

    # Дублируем времена в днях для удобства
    print(f"\nВ ЕДИНИЦАХ «ДЕНЬ» (для наглядности):")
    print(f"  T_рем  = {fmt_fin(r['T_repair'] * 30)} дней (= t = {t_days})")
    print(f"  T_ож   = {fmt_fin(r['T_wait'] * 30)} дней "
          f"(≈ {fmt_fin(r['T_wait'] * 30 * 24)} часов)")
    print(f"  T_сис  = {fmt_fin(r['T_sys'] * 30)} дней")
    print(f"\nАбсолютная пропускная способность в днях:")
    print(f"  A = {fmt_fin(r['A'] / 30)} заявок/день "
          f"= {fmt_fin(r['A'])} заявок/мес")

    # ---------- Графики ----------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    g1 = os.path.join(out_dir, "task5_state_distribution.png")
    g2 = os.path.join(out_dir, "task5_prob_vs_threshold.png")
    plot_state_distribution(r["P"], n, i_max, P_percent, g1)
    plot_prob_vs_threshold(r["P"], n, P_percent, g2)
    print(f"\nГрафики сохранены:")
    print(f"  • {g1}")
    print(f"  • {g2}")

    # ---------- Выводы ----------
    print("\n" + "=" * 72)
    print("  ВЫВОДЫ")
    print("=" * 72)
    print(
        f"1. Нагрузка системы невелика: ρ = {fmt_mid(rho)} per source, "
        f"n·ρ = {fmt_mid(n * rho)}. Большую часть времени все источники "
        f"активны (P_0 = {fmt_fin(r['P'][0])}, {fmt_fin(r['P'][0] * 100)} %)."
    )
    print(
        f"2. Вероятность того, что не менее {P_percent} % источников "
        f"активны, составляет {fmt_fin(prob)} ({fmt_fin(prob * 100)} %). "
        f"То есть событие практически достоверное — в системе с большим "
        f"запасом по «ремонтной способности»."
    )
    print(
        f"3. Среднее число неисправных источников L_неис = "
        f"{fmt_fin(r['L_neis'])} из {n}; на ремонте в среднем "
        f"{fmt_fin(r['L_repair'])} (≈ загрузка канала ремонта). Очередь "
        f"короткая: L_оч = {fmt_fin(r['L_queue'])}."
    )
    print(
        f"4. Абсолютная пропускная способность A = {fmt_fin(r['A'])} "
        f"заявок/мес — практически равна заданному входу k = {k_total} "
        f"заявок/мес. Это самосогласованность модели: установившийся поток "
        f"ремонтов = установившийся поток поломок."
    )
    print(
        f"5. Среднее время ожидания ремонта T_ож = "
        f"{fmt_fin(r['T_wait'] * 30)} дней ({fmt_fin(r['T_wait'] * 30 * 24)} "
        f"часов) — заметно меньше самого ремонта T_рем = {t_days} дней. "
        f"Полное время «в неисправности» T_сис = "
        f"{fmt_fin(r['T_sys'] * 30)} дней."
    )
    print(
        f"6. Влияние параметров: при росте λ (например, удвоение k) или "
        f"росте t (медленный ремонт) ρ увеличивается, очередь удлиняется и "
        f"доля времени «много сломанных» растёт. Чтобы поддерживать долю "
        f"активных ≥ {P_percent} %, нужно либо ускорять ремонт (уменьшать "
        f"t), либо добавлять ремонтные каналы."
    )


if __name__ == "__main__":
    main()
