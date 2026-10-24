# Usage: python visualize.py my_policy.py --seed 9001 [--out episode.png]   (needs matplotlib)
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sim import load_scenario, run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--out", default=None, help="save to this file instead of opening a window")
    args = ap.parse_args()

    try:
        import matplotlib
        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        sys.exit("visualize.py needs matplotlib:  pip install matplotlib")
    import numpy as np

    scn = load_scenario(args.seed)
    r = run_episode(args.policy, scn, isolate=True, record_trace=True)
    tr = r["trace"]

    fig, ax = plt.subplots(figsize=(9, 8))
    prior = np.asarray(scn.prior)
    ax.imshow(prior.T, origin="lower", extent=(0, 400, 0, 400), cmap="Greys", alpha=0.55)
    seg = np.stack([tr[:-1, 1:3], tr[1:, 1:3]], axis=1)
    lc = LineCollection(seg, cmap="plasma", linewidths=1.6)
    lc.set_array(tr[1:, 3])
    lc.set_clim(5, 80)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label="altitude (m)", fraction=0.046)

    det = r["detections_log"]
    if det:
        hi = [(x, y) for (_, x, y, tri) in det if tri is None]
        lo = [(x, y) for (_, x, y, tri) in det if tri is not None]
        if hi:
            ax.scatter(*zip(*hi), s=10, marker="x", c="tab:blue", lw=0.8, label="detection (no triage)")
        if lo:
            ax.scatter(*zip(*lo), s=12, marker="+", c="tab:cyan", lw=0.8, label="detection (with triage)")
    good = [(d["x"], d["y"]) for d in r["declarations"] if d["correct"]]
    bad = [(d["x"], d["y"]) for d in r["declarations"] if not d["correct"]]
    if good:
        ax.scatter(*zip(*good), s=70, facecolors="none", edgecolors="green", lw=2, label="declaration (correct)")
    if bad:
        ax.scatter(*zip(*bad), s=70, marker="X", c="red", label="declaration (false)")
    if scn.is_calibration:
        colors = {"critical": "red", "serious": "orange", "minor": "gold"}
        for (x, y), tri in zip(scn.survivors, scn.triage):
            ax.scatter([x], [y], s=110, marker="*", c=colors[tri], edgecolors="k", lw=0.6, zorder=5)
        ax.scatter([], [], s=110, marker="*", c="orange", edgecolors="k", label="true survivor (calibration)")
    ax.scatter([0], [0], s=90, marker="s", c="k", label="base")
    ax.set_xlim(-5, 405)
    ax.set_ylim(-5, 405)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("%s  score %+.3f  value %.3f  false %d  landed %s  (%s)" % (
        scn.scenario_id, r["score"], r["value_norm"], r["n_false"], r["landed"], r["end_reason"]))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=130)
        print("saved", args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
