"""Campaign cost estimate for THIS laptop, from measured throughput."""
from omegaconf import OmegaConf

c = OmegaConf.load("config.yaml")
H = len(c.heuristics.pool)                      # 6
N = int(round(c.sim.shift_hours * 60 / c.sim.interval_minutes))   # 32
M = int(c.labeling.n_rollout_samples)           # 20
TAU = int(c.labeling.tau)                       # 4
NTR, NTE = int(c.shifts.n_train), int(c.shifts.n_test)
e4 = c.experiments.e4_sensitivity
e10 = c.experiments.e10_misspecification
e2 = c.experiments.e2

# Measured on this machine (AMD Ryzen 9 7940HS, 8 physical / 16 logical).
RATE_1 = 102.0     # interval-steps/s, single worker
RATE_N = 1025.0    # interval-steps/s, n_jobs=-1  -> 10.0x effective

def label(shifts, tau, m=M, h=H):
    return shifts * (N + N * h * m * tau)

def eval_static(methods, shifts=NTE):
    return methods * shifts * N

def eval_mpc(tau, shifts=NTE, m=M, h=H):
    return shifts * N * h * m * tau

rows = []
def add(stage, steps, note="", train_min=0.0):
    rows.append((stage, steps, train_min, note))

# ---- Stage 2 ----
add("2  label corpus (tau=4, M=20)", label(NTR + NTE, TAU))
add("2  tau=1 arm (make tau1)", label(NTR + NTE, 1))

# ---- Stage 3 ---- (no simulation; GMM K-sweep + 18-config x 5-fold XGBoost)
add("3  regime + ranker + calibrator", 0, "GMM sweep + 18x5 CV", train_min=25)
add("3  tau=1 ranker", 0, "--skip-cv-cal", train_min=4)

# ---- Stage 4 ----
s4 = eval_static(13) + eval_mpc(TAU) + eval_mpc(1)
s4 += 8_000 + 12 * 8_000          # ppo_fair + the 12-config PPO sensitivity
                                  # (measured: ~36 s/config, ~7 min total)
add("4  baselines + RL sensitivity", s4,
    "rolling_mpc measured at ~32 s/shift; run it with --n-jobs -1",
    train_min=25)   # PPO x21 runs + FQI 12-config hpsearch

# ---- Stage 4b: sample-efficiency curves ----
n_de = len(e2.data_efficiency_budgets) * int(e2.data_efficiency_reps)   # 20
add("4b data-efficiency: DAHS", eval_static(n_de), f"{n_de} retrains", train_min=n_de * 0.5)
add("4b data-efficiency: offline_fqi", eval_static(n_de), f"{n_de} retrains", train_min=n_de * 0.5)
add("4b e9 robustness_grid (frozen)", eval_static(12))

# ---- Stage 5 ----
per_scenario = eval_mpc(TAU) + eval_mpc(1) + eval_static(15)
add("5  scenarios (balanced + high_load_perish)", 2 * per_scenario)
add("5  e8 robustness grid (12 cells x 4 methods)",
    12 * (eval_mpc(1) + eval_static(3)))

n_mis = sum(len(v) for v in e10.values())        # cells across all five axes
add("5  misspecification (5 axes)",
    n_mis * (eval_mpc(TAU) + eval_static(4)), f"{n_mis} cells, incl. rolling_mpc")

n_w = sum(len(e4[a]) for a in ("w_breach", "w_spoil", "w_tardy", "w_holding"))
add("5  objective-weight sweep", n_w * (eval_mpc(TAU) + eval_static(3)), f"{n_w} settings")

add("5  t_min / theta / arrival_noise sweeps",
    (len(e4.t_min) + len(e4.theta) + len(e4.arrival_noise)) * eval_static(1))

# tau sweep: tau=1 already built; needs a LABELLING pass at tau=2 and 3
add("5  tau sweep (relabel tau=2,3 + retrain)",
    label(NTR + NTE, 2) + label(NTR + NTE, 3), "the expensive sensitivity",
    train_min=2 * 20)

add("5  e5 reliability + SHAP, feature/observability/saturation", eval_static(4),
    "SHAP is the slow part", train_min=20)
add("5b real-data (Olist): fit + validate + bursty rerun",
    eval_static(4) + eval_mpc(1), "needs the Olist dataset")

# ---- Ablations ----
add("A  4 retrain ablations + inference ablations", eval_static(8),
    "no relabel", train_min=4 * 20)
add("A  relabel single_sample_rollout (M=1)", label(NTR + NTE, TAU, m=1),
    "M=1 is 1/20 the cost", train_min=20)

# ---- Report ----
w = 52
print(f"{'stage':<{w}}{'Msteps':>9}{'sim h':>8}{'train h':>9}{'total h':>9}")
print("-" * (w + 35))
tot_s = tot_t = 0.0
for stage, steps, tmin, note in rows:
    sim_h = steps / RATE_N / 3600
    tr_h = tmin / 60
    tot_s += sim_h; tot_t += tr_h
    print(f"{stage:<{w}}{steps/1e6:>9.2f}{sim_h:>8.2f}{tr_h:>9.2f}{sim_h+tr_h:>9.2f}")
print("-" * (w + 35))
print(f"{'TOTAL':<{w}}{sum(r[1] for r in rows)/1e6:>9.2f}{tot_s:>8.2f}{tot_t:>9.2f}{tot_s+tot_t:>9.2f}")
print()
print(f"  measured: {RATE_1:.0f} steps/s single-core, {RATE_N:.0f} steps/s at n_jobs=-1 "
      f"({RATE_N/RATE_1:.1f}x effective on 8 physical cores)")
print(f"  ideal 16-core machine would be ~{16/ (RATE_N/RATE_1):.1f}x faster on the sim terms")
base = tot_s + tot_t
print(f"\n  no throttling      : {base:.1f} h")
print(f"  -20% sustained clk : {base/0.8:.1f} h   <- realistic for a 13in laptop chassis")
print(f"  -35% sustained clk : {base/0.65:.1f} h   <- hot room / battery / other load")
