# Find the survivors before the battery dies

A building has collapsed and people are trapped. You control one search drone
that starts at a base in the corner of a 400 m x 400 m area, with one battery.

**Your job:** find the survivors, tell the rescue team where each one is and how
badly hurt they are, and get back to base before the battery dies.

You write the drone's brain: **one Python file with one class, `Policy`.**
We run it on about 100 hidden scenarios and rank submissions by mean score.

---

## 1. Setup

Python **3.10** is required (the simulator only runs on this version).

```bash
python3.10 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
# Windows: py -3.10 -m venv .venv  then  .venv\Scripts\activate  then  pip install -r requirements.txt
# conda:   conda create -n drone python=3.10 -y && conda activate drone && pip install -r requirements.txt
```

Tested on Linux. On Windows, WSL is recommended if anything misbehaves.

## 2. What's in the package

| File | Use |
|---|---|
| `policy_template.py` | Starting point. Copy it to `<rollnumber>.py`. |
| `run_local.py` | Runs your policy on the local scenarios and prints scores. |
| `visualize.py` | Draws one flight: path, altitude, detections, declarations (needs `pip install matplotlib`). |
| `baselines/` | `random_policy.py` and `lawnmower_policy.py`, reference points to beat. |
| `sim/` | The physics simulator. You don't need to read or change it. Scoring is in `sim/scoring.py`. |
| `scenarios/seeds.json` | Seeds of the 20 public and 3 calibration scenarios. |

## 3. What you build

```python
class Policy:
    def __init__(self, info): ...   # called once at the start of each episode
    def act(self, obs): ...         # called before every action; return ONE action dict
```

### The world
- Arena: `x, y` in `[0, 400]` m. Base at `(0, 0)`. The drone starts there at 5 m altitude.
- Prior map: a 40 x 40 grid of 10 m cells. `prior_map[ix][iy]` is the probability
  that survivors are in the cell `x ∈ [10·ix, 10·ix+10)`, `y ∈ [10·iy, 10·iy+10)`.
- Sensor: sees a circle under the drone with radius `altitude × tan(40°)`.
- Survivors have a triage level: `critical`, `serious` or `minor`. Their number is not given.

### Actions (`act()` returns one of these)

| Action | Effect |
|---|---|
| `{"type": "fly_to", "x", "y", "altitude", "speed"}` | Straight line to `(x, y)`. Altitude `[5, 80]` m changes linearly along the way. Speed `[1, 15]` m/s. Vertical speed is at most 4 m/s, so large altitude changes slow the move. |
| `{"type": "hover", "duration"}` | Hover for `[0.5, 30]` s. |
| `{"type": "declare", "x", "y", "triage"}` | Report a survivor. Timestamped when sent, then the drone hovers 1 s. **No feedback on correctness.** |
| `{"type": "land"}` | Land. Only within 15 m of base. Ends the episode. |

Out-of-range values are clamped (status `"clamped"`). Malformed actions, or `land`
away from base, do nothing (status `"invalid"`); 50 invalid actions end the episode.

### `info` (given once)
`arena_size`, `grid_size`, `cell_size`, `base_xy`, `prior_map`, `energy_budget`,
`altitude_range`, `speed_range`, `hover_range`, `max_climb_rate`, `start_altitude`,
`footprint_tan`, `land_radius`, `declare_duration`, the scoring constants
(`triage_weights`, `triage_decay_tau`, `wrong_triage_factor`, `match_radius`,
`lambda_fp`, `return_penalty`), the limits (`max_actions`, `act_time_limit_s`,
`episode_decision_time_limit_s`), and `is_calibration`.

Not given: the number of survivors, the wind, or any physics parameter.

### `obs` (before the first action and after every action)
| Key | Meaning |
|---|---|
| `t`, `x`, `y`, `altitude` | time (s) and position |
| `energy_remaining`, `energy_used_last_action` | battery |
| `last_action_status` | `"ok"`, `"clamped"` or `"invalid"` |
| `detections` | list of `{"x", "y", "triage", "t"}` from the last action; `triage` may be `None` |
| `n_declared`, `n_actions` | counters |
| `calibration_truth` | only in calibration scenarios (section 6) |

A detection is a noisy report of something that looks like a person. The same
survivor can be reported many times, and some reports are not real.

### The episode ends when
you land, the battery hits 0, you reach 1500 actions or 50 invalid actions,
your code raises an exception, or it breaks a time limit. Declarations made
before the end always count.

## 4. Scoring

A declaration is **correct** if it is within **5 m** of a survivor that hasn't
already been claimed (declarations are matched in the order you make them).
Anything else is a **false declaration**.

```
value of a correct declaration = weight × exp(−t / tau) × (1 if triage right, else 0.5)

value_norm    = sum of values / sum of weights of ALL survivors
false_term    = 1.0 × n_false / n_survivors
return_term   = 0.3 if the drone did not land at base, else 0
episode_score = max(−1, value_norm − false_term − return_term)
final score   = mean episode_score over all hidden scenarios
```

| Triage | weight | tau |
|---|---:|---:|
| critical | 3 | 300 s |
| serious | 2 | 600 s |
| minor | 1 | 1200 s |

**Example:** 4 survivors (critical, serious, minor, minor; total weight 7).
You declare the critical one correctly at t = 120 s (3·e^−0.4 = 2.011), the serious
one at t = 200 s but as "minor" (2·e^−1/3·0.5 = 0.717), one empty spot (false), and a
minor one correctly at t = 400 s (1·e^−1/3 = 0.717).
value_norm = 3.444 / 7 = 0.492; false term = 1/4 = 0.25.
Landed: **0.242**. Battery ran out: 0.242 − 0.3 = **−0.058**.

## 5. What you know about the environment

The physics formulas and constants are not given. Discover them by experiment.

- Detection becomes less reliable at higher altitude, and flying fast gives less time over each spot.
- False detections happen, more often higher up, and some areas are more cluttered than others.
- Reported positions are noisier from higher up.
- Triage level can only be seen when close to the ground.
- Energy use depends on heading as well as distance; the environment applies a
  horizontal disturbance whose strength and direction vary by scenario and are
  not reported. It may not stay constant.
- The damage-map prior is informative but imperfect.
- **The evaluation environment has the same physical structure as your local
  simulator but different parameter values and wider scenario ranges.
  Constants measured locally will not transfer exactly.**

## 6. Calibration scenarios

Seeds **9001, 9002, 9003** are your test bench. Only there, every `obs` also contains:

```python
obs["calibration_truth"] = {"survivors": [{"x", "y", "triage"}, ...], "wind": [wx, wy]}
```

Use them to study how the sensor and the battery behave. They never appear in
the evaluation, so your submitted policy must not depend on them.

## 7. Rules

- One file named `<rollnumber>.py` with a class `Policy`.
- Python 3.10. Allowed imports: `numpy`, `scipy`, `math`, `random`, `collections`,
  `heapq`, `itertools`, `functools`, `dataclasses`, `typing`, `bisect`.
- No file access, network, subprocesses, or inspecting the environment.
- Time limits:
  - each `act()`: **200 ms**
  - all `act()` calls in one episode: **30 s**
  - import + `__init__`: **10 s**

  Breaking a limit or raising an exception ends the episode. Declarations
  made so far count, and you get the not-landed penalty.
- Each episode runs in a fresh process; nothing carries over between episodes.
- Be deterministic: seed any randomness you use.

## 8. Running locally

```bash
python run_local.py my_policy.py                          # 20 public scenarios
python run_local.py my_policy.py --scenarios calibration  # 3 calibration scenarios
python run_local.py my_policy.py --seed 1007 --verbose    # one scenario, each declaration + your prints
python visualize.py my_policy.py --seed 9001 --out ep.png # plot one flight
```

`run_local.py` applies the same time limits as the evaluation.

## 9. Baselines (20 public scenarios)

| Policy | Mean score | Mean value_norm | False declarations / scenario | Landed |
|---|---:|---:|---:|---:|
| `random_policy.py` | -0.712 | 0.000 | 2.00 | 0% |
| `lawnmower_policy.py` | +0.306 | 0.417 | 0.60 | 100% |
| `policy_template.py` | +0.000 | 0.000 | 0.00 | 100% |

Evaluation scores will be different (section 5).

## 10. Submitting

Submit only `<rollnumber>.py`. Check first that
`python run_local.py <rollnumber>.py --scenarios all` runs without crashes or timeouts.
