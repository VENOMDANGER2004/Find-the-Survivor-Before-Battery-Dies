# Usage: python run_local.py my_policy.py [--scenarios public|calibration|all] [--seed N] [--verbose]
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sim import load_scenario, run_episode, scenario_seeds


def _run(args):
    policy, seed, is_cal, verbose = args
    res = run_episode(policy, load_scenario(seed, is_cal), isolate=True, show_output=verbose)
    return res


def _fmt(r):
    return ("%-16s score %+.3f  value %.3f  correct %2d/%-2d decl  false %2d  survivors %2d  "
            "landed %-3s  end %-18s t %6.1f  energy_left %6.1f  max_ms %6.1f" % (
                r["scenario_id"], r["score"], r["value_norm"], r["n_correct"], r["n_declared"],
                r["n_false"], r["n_survivors"], "yes" if r["landed"] else "no", r["end_reason"],
                r["t_end"], r["energy_left"], r["decision_ms_max"]))


def main():
    ap = argparse.ArgumentParser(description="Run a policy on the local scenarios with the official time limits.")
    ap.add_argument("policy")
    ap.add_argument("--scenarios", default="public", choices=["public", "calibration", "all"])
    ap.add_argument("--seed", type=int, default=None, help="run a single scenario seed")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--json", default=None, help="also write per-episode results to this JSON file")
    args = ap.parse_args()

    if not os.path.isfile(args.policy):
        sys.exit("no such file: %s" % args.policy)
    seeds = scenario_seeds(args.scenarios)
    if args.seed is not None:
        seeds = [s for s in scenario_seeds("all") if s[0] == args.seed] or [(args.seed, False)]
    jobs = [(args.policy, s, c, args.verbose) for s, c in seeds]

    results = []
    if args.verbose or args.workers <= 1 or len(jobs) == 1:
        for j in jobs:
            r = _run(j)
            results.append(r)
            print(_fmt(r), flush=True)
            if args.verbose:
                for d in r["declarations"]:
                    print("      t=%6.1f  (%6.1f, %6.1f)  %-8s -> %s" % (
                        d["t"], d["x"], d["y"], d["triage"],
                        ("correct, true triage %s, value %.3f" % (d["true_triage"], d["value"]))
                        if d["correct"] else "FALSE"))
            if r["error"]:
                print("      error:", r["error"])
            if args.verbose and r.get("traceback"):
                print(r["traceback"].rstrip())
    else:
        with mp.Pool(args.workers) as pool:
            for r in pool.imap(_run, jobs):
                results.append(r)
                print(_fmt(r), flush=True)
                if r["error"]:
                    print("      error:", r["error"])

    n = len(results)
    mean = lambda k: sum(float(r[k]) for r in results) / n
    print("-" * 100)
    print("mean score %+.4f over %d scenarios | value_norm %.3f | false decl/scenario %.2f | "
          "landed %d/%d | timeouts %d | crashes %d | max act() %.1f ms" % (
              mean("score"), n, mean("value_norm"), mean("n_false"), sum(r["landed"] for r in results), n,
              sum(r["timeout"] for r in results), sum(r["crash"] for r in results),
              max(r["decision_ms_max"] for r in results)))
    if args.json:
        keep = ("scenario_id", "score", "value_norm", "n_false", "n_correct", "n_declared", "n_survivors",
                "landed", "end_reason", "timeout", "crash", "error", "decision_ms_max")
        with open(args.json, "w") as f:
            json.dump([{k: r[k] for k in keep} for r in results], f, indent=1)


if __name__ == "__main__":
    main()
