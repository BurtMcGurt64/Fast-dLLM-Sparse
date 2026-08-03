"""
Plot the batched token-policy results (GSM8K, uniform batching, l2 metric, freeze=3).
Data hardcoded from run_policy_eval.py sweeps. Produces policy_results.png.

Run anywhere with matplotlib:  python plot_policy_results.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- batch sweep: uniform, n=200, l2 thr=0.02, freeze=3 (padded to 501 tok) ---
bs      = [8,    16,   32,   64,   128,  256]
spd_bs  = [1.10, 1.14, 1.20, 1.38, 1.43, 1.52]
skip_bs = [53,   53,   53,   53,   47,   43]
org_bs  = [0.720,0.760,0.720,0.695,0.720,0.730]
pol_bs  = [0.655,0.660,0.675,0.650,0.610,0.605]

# --- threshold sweep: uniform, n=200, bs=64, l2, freeze=3 ---
thr      = [0.005, 0.01, 0.02, 0.05]
spd_thr  = [1.23,  1.21, 1.39, 1.37]
skip_thr = [41,    46,   53,   60]
org_thr  = [0.695, 0.695,0.695,0.695]
pol_thr  = [0.650, 0.690,0.650,0.635]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# panel 1: batch -> speedup (+ skip%)
a = ax[0]
a.plot(bs, spd_bs, "o-", color="tab:blue", lw=2, label="speedup (pol / ORG)")
a.axhline(1.0, color="gray", ls=":", lw=1)
a.set_xscale("log", base=2); a.set_xticks(bs); a.set_xticklabels(bs)
a.set_xlabel("batch size"); a.set_ylabel("wall-clock speedup", color="tab:blue")
a.set_title("Speedup grows with batch\n(compute-bound regime)")
a2 = a.twinx()
a2.plot(bs, skip_bs, "s--", color="tab:orange", lw=1.5, alpha=.7, label="skip%")
a2.set_ylabel("skip% (frozen in ALL seqs)", color="tab:orange"); a2.set_ylim(0, 70)
a.text(64, 1.38, "1.52x @ bs=256", fontsize=9)

# panel 2: batch -> accuracy (ORG vs pol) — the real quality cost
a = ax[1]
a.plot(bs, org_bs, "o-", color="black", lw=2, label="ORG (dense)")
a.plot(bs, pol_bs, "o-", color="tab:red", lw=2, label="token policy")
a.fill_between(bs, pol_bs, org_bs, color="tab:red", alpha=.1)
a.set_xscale("log", base=2); a.set_xticks(bs); a.set_xticklabels(bs)
a.set_xlabel("batch size"); a.set_ylabel("GSM8K accuracy")
a.set_ylim(0.55, 0.80); a.legend(loc="lower left")
a.set_title("Quality cost is real & GROWS at high batch\n(gap ~0.05 @ bs<=64 -> ~0.12 @ bs=256)")

# panel 3: threshold Pareto (bs=64) — speedup vs accuracy
a = ax[2]
sc = a.scatter(spd_thr, pol_thr, c=skip_thr, cmap="viridis", s=140, zorder=3)
for x, y, t in zip(spd_thr, pol_thr, thr):
    a.annotate(f"thr={t}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
a.axhline(0.695, color="black", ls="--", lw=1.5, label="ORG (dense) = 0.695")
a.fill_between([1.15, 1.45], 0.695-0.04, 0.695+0.04, color="gray", alpha=.12,
               label="n=200 noise floor (+-0.04)")
a.set_xlabel("wall-clock speedup (bs=64)"); a.set_ylabel("GSM8K accuracy")
a.set_title("Threshold Pareto @ bs=64\n(thr=0.01 near-lossless @ 1.21x)")
a.legend(loc="lower right", fontsize=8)
plt.colorbar(sc, ax=a, label="skip%")

plt.tight_layout()
plt.savefig("policy_results.png", dpi=150, bbox_inches="tight")
print("wrote policy_results.png")
