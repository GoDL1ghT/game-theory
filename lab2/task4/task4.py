"""
Лабораторная работа по теории массового обслуживания.

Задача 4. СМО с ограничением на время ожидания в очереди
         («нетерпеливые» заявки, модель M/M/k+M).

Расчёт по формулам учебного пособия
М. А. Плескунова «Теория массового обслуживания» (2022).

МОДЕЛЬ:
    k каналов; очередь неограниченной длины; у каждой заявки в
    очереди есть случайное «терпение» — экспоненциально распределённое
    с математическим ожиданием ω. Если ожидание превышает терпение,
    заявка уходит из очереди необслуженной.

    Эквивалентная формулировка через процесс гибели и размножения:
        интенсивность поступления заявок:        λ_i = λ
        интенсивность обслуживания в состоянии i:
            μ_i = i·μ                  при 0 ≤ i ≤ k
            μ_i = k·μ + (i-k)·γ        при i > k,   γ = 1/ω

ИСХОДНЫЕ ДАННЫЕ:
    λ = 1.5    — интенсивность входного потока, заявок/мин
    t = 1      — среднее время обслуживания, мин
    k = 4      — число каналов
    ω = 6      — среднее время «терпения», мин
    C = 120    — средний доход от одной обслуженной заявки, у.е.
    ε = 0.001  — требуемая точность (остаток бесконечного ряда)

ТРЕБУЕТСЯ:
    1) Предельные вероятности состояний;
       вероятность обслуживания заявки;
       L_оч, L_сист; T_оч, T_обс, T_сист.
    2) Средние потери дохода: Д_потерь = C · ν_уход.

ЗАМЕЧАНИЕ К ФОРМУЛЕ ПОТЕРЬ:
    В оригинальном задании указано «ν_уход = ω · L_оч». Это
    размерно некорректно: ν_уход — интенсивность ухода (заявок/мин),
    а ω — время (мин). Согласно стандартному определению,
        интенсивность ухода одной заявки = γ = 1/ω,
    тогда полная интенсивность ухода из очереди:
        ν_уход = γ · L_оч = L_оч / ω,
    что используется ниже.
"""

import math
import os
import matplotlib.pyplot as plt


PREC_MID = 5
PREC_FIN = 3
EPS = 0.001          # требуемая точность хвоста ряда


def fmt_mid(x): return f"{x:.{PREC_MID}f}"
def fmt_fin(x): return f"{x:.{PREC_FIN}f}"
def fmt_sci(x):
    if x == 0 or x >= 1e-4:
        return fmt_mid(x)
    return f"{x:.3e}"


# ----------------------------------------------------------------------
# Расчётные функции
# ----------------------------------------------------------------------

def compute_probabilities(lam: float, mu: float, k: int, omega: float,
                          eps: float) -> tuple:
    """
    Итеративный расчёт предельных вероятностей P_i для M/M/k+M.

    Алгоритм:
      • вводим ненормированные q_i: q_0 = 1, q_i = q_{i-1} · λ / μ_i;
      • суммируем, пока вклад q_i не станет малым: q_i < eps·Σq / 10
        (это гарантирует, что хвост ряда меньше eps);
      • нормируем: P_i = q_i / Σq.

    Возвращает (P, n_states, tail_bound).
    """
    gamma = 1.0 / omega
    q = [1.0]                    # q_0 = 1 (ненормированное)
    s = 1.0                      # текущая частичная сумма
    max_states = 500             # защита от зацикливания
    while len(q) < max_states:
        i = len(q)               # вычисляем q_i
        if i <= k:
            mu_i = i * mu        # i каналов заняты, ничего не уходит
        else:
            mu_i = k * mu + (i - k) * gamma
        q_i = q[-1] * lam / mu_i
        q.append(q_i)
        s += q_i
        # Критерий сходимости: вклад «нового» состояния мал
        if i > k and q_i < eps * s * 0.1:
            break
    # Оценка остатка ряда: хвостовая геометрия с коэффициентом
    # r = λ/(k·μ + (i+1)·γ) << 1
    r_tail = lam / (k * mu + (len(q) - k + 1) * gamma)
    tail_bound = q[-1] * r_tail / (1 - r_tail) / s

    # Нормировка
    P = [qi / s for qi in q]
    return P, len(P), tail_bound


def mmk_impatient_characteristics(lam: float, mu: float, k: int,
                                   omega: float, eps: float) -> dict:
    """
    Полный расчёт характеристик СМО с нетерпеливыми заявками.

    Используются формулы:
       γ = 1/ω
       ρ = λ/μ
       L_оч  = Σ_{i>k} (i-k) · P_i
       L_кан = Σ_{i=1..k} i · P_i + k · Σ_{i>k} P_i
                                              (среднее число занятых каналов)
       L_сис = L_оч + L_кан
       ν_уход = γ · L_оч                       (полная интенсивность ухода)
       P_отк = ν_уход / λ                      (вероятность неполучения услуги)
       P_обс = 1 − P_отк                       (вероятность обслуживания)
       λ_эфф = λ · P_обс                       (эффективный поток)
       T_оч  = L_оч  / λ                       (среднее время ожидания)
       T_обс = 1 / μ                           (среднее время обслуживания)
       T_сис = L_сис / λ                       (среднее время в системе)
    """
    gamma = 1.0 / omega
    rho = lam / mu
    chi = rho / k

    # Предельные вероятности
    P, n_states, tail = compute_probabilities(lam, mu, k, omega, eps)

    # L_оч — среднее число заявок в очереди
    L_q = sum((i - k) * P[i] for i in range(k + 1, n_states))

    # Среднее число занятых каналов
    L_busy = (sum(i * P[i] for i in range(1, min(k + 1, n_states)))
              + k * sum(P[i] for i in range(k + 1, n_states)))

    # Среднее число заявок в системе
    L_sys = L_q + L_busy

    # Интенсивность ухода из очереди
    nu_leave = gamma * L_q                  # = L_q / ω

    # Вероятности
    P_lost = nu_leave / lam
    P_serve = 1.0 - P_lost
    lam_eff = lam * P_serve

    # Времена (формула Литтла по полному входному потоку)
    T_q = L_q / lam                          # ожидания в очереди
    T_serv = 1.0 / mu                         # обслуживания
    T_sys = L_sys / lam                       # пребывания в системе

    return {
        "rho": rho, "chi": chi, "gamma": gamma,
        "P": P, "n_states": n_states, "tail_bound": tail,
        "L_q": L_q, "L_busy": L_busy, "L_sys": L_sys,
        "nu_leave": nu_leave, "P_lost": P_lost, "P_serve": P_serve,
        "lam_eff": lam_eff,
        "T_q": T_q, "T_serv": T_serv, "T_sys": T_sys,
    }


# ----------------------------------------------------------------------
# Функции вывода
# ----------------------------------------------------------------------

def print_state_table(P: list, k: int) -> None:
    """Таблица всех вычисленных состояний."""
    print(f"  {'Состояние':<14} | {'P_i':>13} | Комментарий")
    print(f"  {'-' * 14}-+-{'-' * 13}-+-{'-' * 30}")
    for i, p in enumerate(P):
        if i == 0:
            comment = "все каналы простаивают"
        elif i < k:
            comment = f"занято {i} канал(ов), очереди нет"
        elif i == k:
            comment = "все k каналов заняты"
        else:
            comment = f"в очереди {i - k} заявок(а)"
        print(f"  P_{i:<12} | {fmt_sci(p):>13} | {comment}")
    s = sum(P)
    print(f"  {'Σ P_i':<14} | {fmt_mid(s):>13} | контроль нормировки")


# ----------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------

def plot_state_distribution(P: list, k: int, out_path: str) -> None:
    """Столбчатая диаграмма вероятностей состояний."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = list(range(len(P)))
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
        if p > max(P) * 0.005:
            label = fmt_mid(p) if p >= 1e-4 else f"{p:.1e}"
            ax.text(i, p + max(P) * 0.01, label, ha="center",
                    va="bottom", fontsize=8)
    ax.set_xlabel("Состояние i (число заявок в системе)")
    ax.set_ylabel("Предельная вероятность P_i")
    ax.set_title(f"Распределение вероятностей состояний СМО M/M/k+M "
                 f"(k = {k}, нетерпеливые заявки)\n"
                 f"Серый — простой; синий — заняты каналы; "
                 f"оранжевый — все каналы заняты; красный — очередь")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_to_omega(lam, mu, k, C, omega_orig, eps, out_path):
    """График зависимости показателей от ω (среднего времени терпения)."""
    omegas = [0.5 + 0.5 * i for i in range(30)]   # от 0.5 до 15 мин
    L_q, P_serve, loss = [], [], []
    for w in omegas:
        r = mmk_impatient_characteristics(lam, mu, k, w, eps)
        L_q.append(r["L_q"])
        P_serve.append(r["P_serve"])
        loss.append(C * r["nu_leave"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(omegas, P_serve, color="seagreen", label="P_обс(ω)")
    ax.plot(omegas, [1 - p for p in P_serve], color="crimson",
            label="P_отк(ω)")
    ax.axvline(omega_orig, color="orange", linestyle=":",
               label=f"ω = {omega_orig} (задано)")
    ax.set_xlabel("Среднее время терпения ω, мин")
    ax.set_ylabel("Вероятность")
    ax.set_title("Вероятности обслуживания и ухода")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(omegas, L_q, color="steelblue", label="L_оч(ω)")
    ax.axvline(omega_orig, color="orange", linestyle=":", alpha=0.7)
    ax.set_xlabel("Среднее время терпения ω, мин")
    ax.set_ylabel("L_оч", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(omegas, loss, color="crimson", linestyle="--",
             label="Д_потерь(ω), у.е./мин")
    ax2.set_ylabel("Потери Д_потерь, у.е./мин", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax.set_title("Длина очереди и потери дохода")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right",
              fontsize=9)

    fig.suptitle(f"Зависимости от среднего времени терпения ω "
                 f"(λ = {lam}, μ = {mu}, k = {k})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Главная программа
# ----------------------------------------------------------------------

def main() -> None:
    # ---------- Исходные данные ----------
    lam = 1.5            # λ — заявок/мин
    t_min = 1            # t — мин/заявка
    k = 4                # число каналов
    omega = 6            # ω — среднее время терпения, мин
    C = 120              # у.е./заявка (доход)
    eps = EPS            # точность остатка ряда

    # ---------- Производные параметры ----------
    mu = 1.0 / t_min     # μ — заявок/мин
    rho = lam / mu
    chi = rho / k
    gamma = 1.0 / omega

    print("ИСХОДНЫЕ ДАННЫЕ И РАСЧЁТНЫЕ ПАРАМЕТРЫ")
    print("-" * 72)
    print(f"  Интенсивность потока:                 λ = {lam} заявок/мин")
    print(f"  Среднее время обслуживания:           t = {t_min} мин")
    print(f"  Число каналов:                        k = {k}")
    print(f"  Среднее время терпения:               ω = {omega} мин")
    print(f"  Доход с обслуженной заявки:           C = {C} у.е.")
    print(f"  Точность остатка ряда:                ε = {eps}")
    print()
    print(f"  Производные параметры:")
    print(f"     μ = 1/t                = {fmt_mid(mu)} заявок/мин")
    print(f"     ρ = λ/μ                = {fmt_mid(rho)}")
    print(f"     χ = ρ/k                = {fmt_mid(chi)}  "
          f"(система устойчива и без нетерпеливости)")
    print(f"     γ = 1/ω                = {fmt_mid(gamma)} 1/мин  "
          f"(интенсивность ухода одной заявки)")

    # ---------- Задание 1. Характеристики СМО ----------
    r = mmk_impatient_characteristics(lam, mu, k, omega, eps)

    print("\n" + "=" * 72)
    print(f"  ЗАДАНИЕ 1. Характеристики работы СМО")
    print("=" * 72)
    print(f"Учтено {r['n_states']} состояний; оценка хвоста ряда: "
          f"{fmt_sci(r['tail_bound'])} (< ε = {eps}).")
    print()
    print("Таблица 1. Предельные вероятности состояний:")
    print_state_table(r["P"], k)

    print()
    print("ОКОНЧАТЕЛЬНЫЕ ХАРАКТЕРИСТИКИ ОБСЛУЖИВАНИЯ (3 знака):")
    print(f"  Вероятность ухода (потери)            P_отк   = "
          f"{fmt_fin(r['P_lost'])}")
    print(f"  Вероятность обслуживания              P_обс   = "
          f"{fmt_fin(r['P_serve'])}")
    print(f"  Среднее число заявок в очереди        L_оч    = "
          f"{fmt_fin(r['L_q'])}")
    print(f"  Среднее число занятых каналов         L_кан   = "
          f"{fmt_fin(r['L_busy'])}")
    print(f"  Среднее число заявок в системе        L_сист  = "
          f"{fmt_fin(r['L_sys'])}")
    print(f"  Среднее время ожидания                T_оч    = "
          f"{fmt_fin(r['T_q'])} мин")
    print(f"  Среднее время обслуживания            T_обс   = "
          f"{fmt_fin(r['T_serv'])} мин")
    print(f"  Среднее время в системе               T_сис   = "
          f"{fmt_fin(r['T_sys'])} мин")
    print(f"  Эффективный поток обслуженных         λ_эфф   = "
          f"{fmt_fin(r['lam_eff'])} заявок/мин")
    print(f"  Интенсивность ухода из очереди        ν_уход  = "
          f"{fmt_fin(r['nu_leave'])} заявок/мин")

    # ---------- Задание 2. Потеря дохода ----------
    loss_per_min = C * r["nu_leave"]
    loss_per_hour = loss_per_min * 60
    loss_per_8h = loss_per_min * 60 * 8

    print("\n" + "=" * 72)
    print(f"  ЗАДАНИЕ 2. Средние потери дохода из-за ухода заявок")
    print("=" * 72)
    print(f"Формула:   Д_потерь = C · ν_уход,   где ν_уход = γ · L_оч "
          f"= L_оч / ω")
    print(f"Расчёт:    ν_уход = {fmt_mid(r['L_q'])} / {omega} = "
          f"{fmt_mid(r['nu_leave'])} заявок/мин")
    print(f"           Д_потерь = {C} · {fmt_mid(r['nu_leave'])} = "
          f"{fmt_mid(loss_per_min)} у.е./мин")
    print()
    print("Потери в разных единицах времени (3 знака):")
    print(f"  За 1 минуту работы:                   {fmt_fin(loss_per_min)} у.е.")
    print(f"  За 1 час работы:                      {fmt_fin(loss_per_hour)} у.е.")
    print(f"  За 8-часовую смену:                   {fmt_fin(loss_per_8h)} у.е.")

    # Сравнительный расчёт по «литеральной» (некорректной по размерности)
    # формуле — приводим для прозрачности.
    nu_literal = omega * r["L_q"]
    loss_literal = C * nu_literal
    print(f"\nСправочно: если буквально применить формулу из условия")
    print(f"           ν_уход = ω·L_оч = {omega}·{fmt_mid(r['L_q'])} = "
          f"{fmt_mid(nu_literal)} (размерность мин·заявок),")
    print(f"           то Д_потерь = {C}·{fmt_mid(nu_literal)} = "
          f"{fmt_mid(loss_literal)} (тех же странных единиц).")
    print(f"           Этот результат физически не интерпретируется.")

    # ---------- Графики ----------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    g1 = os.path.join(out_dir, "task4_state_distribution.png")
    g2 = os.path.join(out_dir, "task4_sensitivity_omega.png")
    plot_state_distribution(r["P"], k, g1)
    plot_sensitivity_to_omega(lam, mu, k, C, omega, eps, g2)
    print(f"\nГрафики сохранены:")
    print(f"  • {g1}")
    print(f"  • {g2}")

    # ---------- Выводы ----------
    print("\n" + "=" * 72)
    print("  ВЫВОДЫ")
    print("=" * 72)
    print(
        f"1. Загрузка системы умеренная: ρ = {fmt_mid(rho)} при k = {k}, "
        f"χ = ρ/k = {fmt_mid(chi)} ({fmt_fin(chi * 100)} %). Очередь "
        f"формируется редко — основная масса вероятности приходится на "
        f"состояния «0–{min(k, r['n_states'] - 1)} заявок в системе»."
    )
    print(
        f"2. Вероятность обслуживания P_обс = {fmt_fin(r['P_serve'])} "
        f"({fmt_fin(r['P_serve'] * 100)} %), вероятность ухода "
        f"P_отк = {fmt_fin(r['P_lost'])} ({fmt_fin(r['P_lost'] * 100)} %). "
        f"При большом терпении (ω = {omega} мин) и быстром обслуживании "
        f"(t = {t_min} мин) большинство заявок успевают получить сервис."
    )
    print(
        f"3. L_оч = {fmt_fin(r['L_q'])} ≪ 1, T_оч = "
        f"{fmt_fin(r['T_q'])} мин — заявки почти не ждут. Среднее время "
        f"в системе T_сис = {fmt_fin(r['T_sys'])} мин лишь немного "
        f"превышает время обслуживания t = {t_min} мин."
    )
    print(
        f"4. Потери дохода ν_уход·C = {fmt_fin(loss_per_min)} у.е./мин "
        f"(≈ {fmt_fin(loss_per_hour)} у.е./ч) пренебрежимо малы по "
        f"сравнению с доходом λ·P_обс·C = "
        f"{fmt_fin(lam * r['P_serve'] * C)} у.е./мин."
    )
    print(
        f"5. Чувствительность (см. task4_sensitivity_omega.png): "
        f"при уменьшении ω (заявки менее терпеливы) L_оч слегка падает, "
        f"но P_отк растёт — и потери увеличиваются. При увеличении ω "
        f"система приближается к классической M/M/k без нетерпеливости. "
        f"Главный рычаг снижения потерь — увеличение μ (ускорение "
        f"обслуживания) либо k (число каналов): и то, и другое снижает "
        f"L_оч и, следовательно, ν_уход."
    )


if __name__ == "__main__":
    main()
