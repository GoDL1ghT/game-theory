"""
Laboratory work No. 3 - Optimization methods & Dynamic Programming.
Variant 3.  Software-engineering context.

The module solves five problems and prints a structured report.
All numerical artefacts are additionally dumped to `results.json`
so that the written report is built from verified numbers only.

Required libraries:
    numpy, scipy.optimize.linprog, scipy.optimize.minimize_scalar,
    scipy.interpolate.interp1d, collections.deque
"""

import json
import numpy as np
from scipy.optimize import linprog, minimize_scalar
from scipy.interpolate import interp1d
from collections import deque

RESULTS = {}                      # collected numerical results -> results.json
np.set_printoptions(precision=4, suppress=True)


# ======================================================================
# TASK 1 - Linear Programming: release planning of micro-services
# ======================================================================
def task1_linear_programming():
    """
    Maximise expected margin  F = c^T x  subject to mixed <=/>= constraints.

    linprog minimises, therefore the objective is negated.
    '>=' rows are converted to '<=' by multiplying by -1.
    Dual (shadow) prices are taken from res.ineqlin.marginals
    (the sensitivity d(min objective)/d(b_ub)).
    """
    profit = np.array([9, 12, 10, 15, 8, 19], dtype=float)   # margin per service type
    c = -profit                                              # negate for minimisation

    # Original constraint matrix (natural senses)
    #   rows 0..2  -> "<="  resource limits
    #   rows 3..4  -> ">="  demand limits
    #   row  5     -> "<="  gateway limit
    A_ub = np.array([
        [3, 4, 2, 5, 2, 6],        # DevOps-hours        <= 220
        [2, 3, 1, 4, 1, 5],        # cloud budget        <= 160
        [15, 20, 10, 25, 8, 30],   # compute quotas      <= 850
        [-1, -1, -1, 0, 0, 0],     # x1+x2+x3 >= 18  -> -(...) <= -18
        [0, 0, 0, -1, -1, -1],     # x4+x5+x6 >= 12  -> -(...) <= -12
        [1, 0, 0, 0, 0, 1],        # x1+x6               <= 14
    ], dtype=float)
    b_ub = np.array([220, 160, 850, -18, -12, 14], dtype=float)
    bounds = [(0, None)] * 6

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    assert res.success, res.message

    x = res.x
    F = float(profit @ x)

    # ----- shadow prices in terms of the ORIGINAL right-hand sides -----
    # dF/db_ub = -marginals.  For the negated '>=' rows db_ub/dRHS = -1.
    m = res.ineqlin.marginals                       # d(min)/d(b_ub)
    sign = np.array([1, 1, 1, -1, -1, 1])           # -1 for the two negated rows
    shadow = -m * sign                              # dF/d(original RHS)
    slack = res.ineqlin.residual                    # b_ub - A_ub x  (>=0)

    names = ["DevOps-hours (<=220)", "Cloud budget (<=160)",
             "Compute quotas (<=850)", "Critical services (>=18)",
             "Infrastructure balance (>=12)", "Gateways x1+x6 (<=14)"]

    # ----- explicit dual LP (verification of strong duality) -----
    # Primal (all converted to <=):  max profit^T x , A_ub x <= b_ub.
    # Dual: min b_ub^T y , A_ub^T y >= profit , y >= 0.
    c_dual = b_ub
    A_dual = -A_ub.T                                # A_ub^T y >= profit  ->  -A_ub^T y <= -profit
    b_dual = -profit
    res_d = linprog(c_dual, A_ub=A_dual, b_ub=b_dual,
                    bounds=[(0, None)] * 6, method="highs")
    dual_obj = float(c_dual @ res_d.x)

    # ----- sensitivity of the cloud budget  b2 = 160 by +/-15% -----
    base_b2 = 160.0
    sweep = []
    for b2 in np.round(np.arange(120, 200 + 0.001, 2.0), 2):   # wide grid for the curve
        bb = b_ub.copy(); bb[1] = b2
        r = linprog(c, A_ub=A_ub, b_ub=bb, bounds=bounds, method="highs")
        sweep.append((float(b2), float(profit @ r.x) if r.success else None))

    pts = {}
    for tag, b2 in [("-15%", base_b2 * 0.85), ("base", base_b2), ("+15%", base_b2 * 1.15)]:
        bb = b_ub.copy(); bb[1] = b2
        r = linprog(c, A_ub=A_ub, b_ub=bb, bounds=bounds, method="highs")
        pts[tag] = {"b2": round(float(b2), 2),
                    "F": round(float(profit @ r.x), 4),
                    "x": [round(v, 4) for v in r.x]}

    RESULTS["task1"] = {
        "x": [round(v, 6) for v in x],
        "F": round(F, 6),
        "constraints": [
            {"name": names[i],
             "lhs": round(float(A_ub[i] @ x), 4),
             "rhs": float(b_ub[i]),
             "slack": round(float(slack[i]), 4),
             "shadow_price": round(float(shadow[i]), 6),
             "binding": bool(abs(slack[i]) < 1e-6)}
            for i in range(6)
        ],
        "dual_objective": round(dual_obj, 6),
        "dual_y": [round(v, 6) for v in res_d.x],
        "sensitivity_points": pts,
        "sensitivity_sweep": sweep,
    }

    # ----- console report -----
    print("=" * 70)
    print("TASK 1 - LINEAR PROGRAMMING")
    print("=" * 70)
    print("Optimal plan  x* =", np.round(x, 4))
    print(f"Optimal margin F* = {F:.4f}  (mln RUB / quarter)")
    print(f"Dual objective    = {dual_obj:.4f}  (strong duality check)")
    print("\n  Constraint                       LHS      RHS    slack   shadow")
    for i in range(6):
        print(f"  {names[i]:<32}{A_ub[i]@x:8.2f}{b_ub[i]:8.1f}"
              f"{slack[i]:8.2f}{shadow[i]:9.4f}")
    print("\n  Cloud-budget sensitivity (+/-15%):")
    for tag in ("-15%", "base", "+15%"):
        p = pts[tag]
        print(f"    {tag:>5}:  b2={p['b2']:6.1f}  ->  F*={p['F']:.4f}")
    return res


# ======================================================================
# TASK 2 - Non-linear DP: load-balancing cost minimisation
# ======================================================================
def task2_nonlinear_dp():
    """
    F1(X)        = min_{0<=Y<=X} [ 5Y^2 + (X-Y)^2 ]
    Fk(X)        = min_{0<=Y<=X} [ 5Y^2 + (X-Y)^2 + F_{k-1}(0.2Y + 0.8(X-Y)) ]
    N = 4 stages, X = 15 TB, grid step h = 1, linear interpolation.
    """
    h, N, X0 = 1.0, 4, 15.0
    Xg = np.arange(0.0, X0 + h, h)             # state grid 0..15

    def stage_cost(Y, X):
        return 5.0 * Y ** 2 + (X - Y) ** 2

    def trans(Y, X):                            # next-stage resource
        return 0.2 * Y + 0.8 * (X - Y)

    F = {}                                      # F[k] : value on the grid
    Yopt = {}                                   # Yopt[k] : optimal split on the grid

    # base stage k = 1
    f1, y1 = np.zeros_like(Xg), np.zeros_like(Xg)
    for i, X in enumerate(Xg):
        if X == 0:
            continue
        r = minimize_scalar(lambda Y: stage_cost(Y, X), bounds=(0, X), method="bounded")
        f1[i], y1[i] = r.fun, r.x
    F[1], Yopt[1] = f1, y1

    # stages k = 2..N use linear interpolation of F[k-1]
    for k in range(2, N + 1):
        Fprev = interp1d(Xg, F[k - 1], kind="linear", fill_value="extrapolate")
        fk, yk = np.zeros_like(Xg), np.zeros_like(Xg)
        for i, X in enumerate(Xg):
            if X == 0:
                continue
            r = minimize_scalar(
                lambda Y: stage_cost(Y, X) + float(Fprev(trans(Y, X))),
                bounds=(0, X), method="bounded")
            fk[i], yk[i] = r.fun, r.x
        F[k], Yopt[k] = fk, yk

    # trajectory restoration for X = 15 (stage 4 -> 1)
    traj, X = [], X0
    for k in range(N, 0, -1):
        if k >= 2:
            Fprev = interp1d(Xg, F[k - 1], kind="linear", fill_value="extrapolate")
            obj = lambda Y: stage_cost(Y, X) + float(Fprev(trans(Y, X)))
        else:
            obj = lambda Y: stage_cost(Y, X)
        r = minimize_scalar(obj, bounds=(0, X), method="bounded")
        Ys = r.x
        traj.append({"stage": k, "X": round(X, 4), "Y": round(Ys, 4),
                     "X_minus_Y": round(X - Ys, 4),
                     "stage_cost": round(stage_cost(Ys, X), 4),
                     "next_X": round(trans(Ys, X), 4)})
        X = trans(Ys, X)

    RESULTS["task2"] = {
        "grid": [round(v, 2) for v in Xg],
        "F": {k: [round(v, 4) for v in F[k]] for k in F},
        "Yopt": {k: [round(v, 4) for v in Yopt[k]] for k in Yopt},
        "F4_at_15": round(float(F[4][-1]), 4),
        "trajectory": traj,
        "traj_total_cost": round(sum(t["stage_cost"] for t in traj), 4),
    }

    print("\n" + "=" * 70)
    print("TASK 2 - NON-LINEAR DP (cost minimisation)")
    print("=" * 70)
    print("  X  | " + " | ".join(f"F{k}" for k in range(1, N + 1)))
    for i, X in enumerate(Xg):
        print(f"{X:4.0f} | " + " | ".join(f"{F[k][i]:8.2f}" for k in range(1, N + 1)))
    print(f"\n  Minimal total cost  F4(15) = {F[4][-1]:.4f}")
    print("  Optimal trajectory (stage: X -> Y, X-Y -> next X):")
    for t in traj:
        print(f"    stage {t['stage']}:  X={t['X']:6.2f}  Y={t['Y']:6.3f}  "
              f"X-Y={t['X_minus_Y']:6.3f}  cost={t['stage_cost']:7.3f}  ->  X'={t['next_X']:6.3f}")
    print(f"    sum of stage costs = {RESULTS['task2']['traj_total_cost']:.4f}")


# ======================================================================
# TASK 3 - Johnson's algorithm: CI/CD conveyor (log processing -> indexing)
# ======================================================================
def task3_johnson():
    """Two-machine flow shop. Johnson's rule minimises the makespan."""
    A = [5, 3, 9, 2, 7, 4, 6, 8]                # stage A : log processing
    B = [2, 6, 4, 8, 1, 5, 3, 7]                # stage B : indexing
    n = len(A)

    def johnson_order(A, B):
        front, back = deque(), deque()
        remaining = set(range(n))
        while remaining:
            # global minimum among remaining A_i and B_i (ties -> machine A)
            best_job, best_val, machine = None, None, None
            for i in sorted(remaining):
                if best_val is None or A[i] < best_val:
                    best_val, best_job, machine = A[i], i, "A"
            for i in sorted(remaining):
                if B[i] < best_val:
                    best_val, best_job, machine = B[i], i, "B"
            if machine == "A":
                front.append(best_job)          # schedule early
            else:
                back.appendleft(best_job)        # schedule late
            remaining.discard(best_job)
        return list(front) + list(back)

    def metrics(order):
        cA = cB = 0
        timeline = []
        for i in order:
            startA = cA
            cA += A[i]                            # machine A is never idle
            startB = max(cB, cA)                  # B waits for A and for itself
            cB = startB + B[i]
            timeline.append((i, startA, cA, startB, cB))
        makespan = cB
        idleB = makespan - sum(B)                 # total idle of machine B
        return makespan, idleB, timeline

    order = johnson_order(A, B)
    ms_j, idle_j, tl = metrics(order)
    order0 = list(range(n))
    ms0, idle0, _ = metrics(order0)

    RESULTS["task3"] = {
        "A": A, "B": B,
        "johnson_order_1based": [i + 1 for i in order],
        "johnson_makespan": ms_j,
        "johnson_idleB": idle_j,
        "original_order_1based": [i + 1 for i in order0],
        "original_makespan": ms0,
        "original_idleB": idle0,
        "timeline": [{"job": i + 1, "A_start": a0, "A_end": a1,
                      "B_start": b0, "B_end": b1} for (i, a0, a1, b0, b1) in tl],
        "sumA": sum(A), "sumB": sum(B),
    }

    print("\n" + "=" * 70)
    print("TASK 3 - JOHNSON'S ALGORITHM")
    print("=" * 70)
    print("  Optimal order      :", [i + 1 for i in order])
    print(f"  Makespan (Johnson) : {ms_j} min,  machine-B idle = {idle_j} min")
    print(f"  Makespan (original): {ms0} min,  machine-B idle = {idle0} min")
    print("  Gantt (job: A[start-end], B[start-end]):")
    for (i, a0, a1, b0, b1) in tl:
        print(f"    job {i+1}:  A[{a0:2d}-{a1:2d}]  B[{b0:2d}-{b1:2d}]")


# ======================================================================
# TASK 4 - Equipment replacement: micro-service / framework rotation
# ======================================================================
def task4_equipment_replacement():
    """
    R(t) = 20 - t   (profit of age-t equipment),
    C(t) = 3 + 4t   (cost of replacing age-t equipment).
    N = 6 years, Tmax = 6, initial age t0 = 3.

    Bellman:
        f_k(t) = max{ KEEP    : R(t)        + f_{k+1}(t+1),       (t < Tmax)
                      REPLACE : R(0) - C(t) + f_{k+1}(1) }
        f_{N+1}(t) = 0.   At t = Tmax keeping is forbidden.
    """
    N, Tmax, t0 = 6, 6, 3
    R = lambda t: 20 - t
    C = lambda t: 3 + 4 * t
    ages = list(range(Tmax + 1))

    f = {N + 1: {t: 0.0 for t in ages}}
    dec = {}
    for k in range(N, 0, -1):
        f[k], dec[k] = {}, {}
        for t in ages:
            keep = R(t) + f[k + 1][t + 1] if t < Tmax else None
            repl = (R(0) - C(t)) + f[k + 1][1]
            if keep is not None and keep >= repl:
                f[k][t], dec[k][t] = keep, "K"
            else:
                f[k][t], dec[k][t] = repl, "R"

    # restore the optimal policy from t0
    policy, t = [], t0
    for k in range(1, N + 1):
        d = dec[k][t]
        gain = R(t) if d == "K" else R(0) - C(t)
        policy.append({"year": k, "age": t, "decision": d,
                       "year_profit": gain, "f_value": round(f[k][t], 4)})
        t = t + 1 if d == "K" else 1

    RESULTS["task4"] = {
        "R": {t: R(t) for t in ages}, "C": {t: C(t) for t in ages},
        "f": {k: {t: round(f[k][t], 4) for t in ages} for k in range(1, N + 1)},
        "decision": {k: dict(dec[k]) for k in range(1, N + 1)},
        "policy": policy,
        "optimal_profit": round(f[1][t0], 4),
        "Tmax": Tmax, "t0": t0, "N": N,
    }

    print("\n" + "=" * 70)
    print("TASK 4 - EQUIPMENT REPLACEMENT")
    print("=" * 70)
    header = "  age |" + "".join(f"  f{k}(t)" for k in range(1, N + 1))
    print(header)
    for t in ages:
        row = f"  {t:3d} |"
        for k in range(1, N + 1):
            row += f" {f[k][t]:5.0f}{dec[k][t]}"
        print(row)
    print(f"\n  Optimal expected profit f1({t0}) = {f[1][t0]:.0f}")
    print("  Optimal policy:")
    for p in policy:
        print(f"    year {p['year']}: age {p['age']:>2}  ->  "
              f"{'KEEP' if p['decision']=='K' else 'REPLACE'}  (profit {p['year_profit']})")


# ======================================================================
# TASK 5 - Resource allocation DP (3 quarters)
# ======================================================================
def task5_resource_allocation():
    """
    g(Y)     = 6Y + 0.15 Y^2          return of activity 1
    h(X-Y)   = 5(X-Y) + 0.1(X-Y)^2    return of activity 2
    carry-over = a*Y + b*(X-Y),  a = 0.4, b = 0.6
    Z = 20, N = 3 quarters, X-grid {0,5,10,15,20}, step dY = 3.

        F3(X) = max_Y [ g(Y) + h(X-Y) ]                     (last quarter)
        Fk(X) = max_Y [ g(Y) + h(X-Y) + F_{k+1}(a Y + b(X-Y)) ]
    """
    Z, N = 20.0, 3
    a, b, dY = 0.4, 0.6, 3.0
    Xg = np.array([0, 5, 10, 15, 20], dtype=float)

    g = lambda Y: 6 * Y + 0.15 * Y ** 2
    hh = lambda z: 5 * z + 0.1 * z ** 2
    trans = lambda Y, X: a * Y + b * (X - Y)

    def best(X, Fnext):
        """Maximise over Y in steps of dY on [0,X] (endpoint X included)."""
        Ys = list(np.arange(0.0, X + 1e-9, dY))
        if not Ys or abs(Ys[-1] - X) > 1e-9:
            Ys.append(X)
        bestv, bestY = -1e18, 0.0
        for Y in Ys:
            v = g(Y) + hh(X - Y)
            if Fnext is not None:
                v += float(Fnext(trans(Y, X)))
            if v > bestv + 1e-12:
                bestv, bestY = v, Y
        return bestv, bestY

    F, Yopt = {}, {}
    f3, y3 = np.zeros_like(Xg), np.zeros_like(Xg)
    for i, X in enumerate(Xg):
        f3[i], y3[i] = best(X, None)
    F[3], Yopt[3] = f3, y3

    for k in (2, 1):
        Fnext = interp1d(Xg, F[k + 1], kind="linear", fill_value="extrapolate")
        fk, yk = np.zeros_like(Xg), np.zeros_like(Xg)
        for i, X in enumerate(Xg):
            fk[i], yk[i] = best(X, Fnext)
        F[k], Yopt[k] = fk, yk

    # restore optimal trajectory from Z (quarter 1 -> 3)
    traj, X = [], Z
    for k in (1, 2, 3):
        Fnext = interp1d(Xg, F[k + 1], kind="linear",
                         fill_value="extrapolate") if k < 3 else None
        v, Ys = best(X, Fnext)
        immediate = g(Ys) + hh(X - Ys)
        traj.append({"quarter": k, "X": round(X, 4), "Y": round(Ys, 4),
                     "X_minus_Y": round(X - Ys, 4),
                     "immediate_return": round(immediate, 4),
                     "next_X": round(trans(Ys, X), 4)})
        X = trans(Ys, X)

    RESULTS["task5"] = {
        "grid": [round(v, 2) for v in Xg],
        "F": {k: [round(v, 4) for v in F[k]] for k in F},
        "Yopt": {k: [round(v, 4) for v in Yopt[k]] for k in Yopt},
        "F1_at_Z": round(float(F[1][-1]), 4),
        "trajectory": traj,
        "traj_total_return": round(sum(t["immediate_return"] for t in traj), 4),
    }

    print("\n" + "=" * 70)
    print("TASK 5 - RESOURCE ALLOCATION DP")
    print("=" * 70)
    print("   X  |   F1     Y1 |   F2     Y2 |   F3     Y3")
    for i, X in enumerate(Xg):
        print(f"  {X:3.0f} | {F[1][i]:6.2f} {Yopt[1][i]:4.0f} |"
              f" {F[2][i]:6.2f} {Yopt[2][i]:4.0f} |"
              f" {F[3][i]:6.2f} {Yopt[3][i]:4.0f}")
    print(f"\n  Maximal total return  F1({Z:.0f}) = {F[1][-1]:.4f}")
    print("  Optimal trajectory:")
    for t in traj:
        print(f"    quarter {t['quarter']}:  X={t['X']:6.2f}  Y={t['Y']:5.2f}  "
              f"X-Y={t['X_minus_Y']:5.2f}  return={t['immediate_return']:7.3f}  ->  X'={t['next_X']:6.3f}")
    print(f"    sum of returns = {RESULTS['task5']['traj_total_return']:.4f}")


if __name__ == "__main__":
    task1_linear_programming()
    task2_nonlinear_dp()
    task3_johnson()
    task4_equipment_replacement()
    task5_resource_allocation()
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, ensure_ascii=False, indent=2)
    print("\n[OK] results.json written")
