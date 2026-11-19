# CD&R testbed with known decision policies (v7.2)

Code for the paper

> J. Belhou, K. Louzaoui, K. Benlhachmi. *A Testbed with Known Decision Policies for Style- and Context-Aware Conflict-Resolution Advisories.* Submitted to the 6th International Conference on Big Data and Artificial Intelligence Applications (ICBDAIA'27).

Simulated air traffic controllers with known decision policies, each defined by a style vector, choose among eight resolution manoeuvres of one aircraft (heading changes of 10°, 20° or 30° to either side, a climb or a descent) in four airspace-and-traffic contexts, under five separation and letter-of-agreement-inspired rules. Recommenders that differ only in their inputs (conflict geometry *X*, controller style *S*, context *C*) learn from the controllers' noisy choices and are scored against each policy's noiseless choice and the exact ceiling of their information.

The controllers are simulated decision policies, not measured people. *Y* is the simulator-generated observed choice (the policy's pick with decision noise; the models learn from *Y* only) and *R\** is the synthetic reference choice (the same policy without noise). Neither is real controller behaviour.

## Files

| File | Content |
|---|---|
| `cdr_testbed_v7_2.py` | The whole experiment in one file: generator, rules and safety filter, decision policies, recommenders, exact ceilings, statistics and tables. It is the exact code of the paper's final run. |

## Requirements

- Python 3.13 (the paper's run used 3.13.5)
- scikit-learn (the paper's run used 1.6.1), numpy, pandas, scipy, joblib, matplotlib
- IPython, optional: tables display as formatted tables in Jupyter or Colab

```bash
pip install scikit-learn==1.6.1 numpy pandas scipy joblib matplotlib
```

## How to run

Run the file as a script (`python cdr_testbed_v7_2.py`) or paste it into one Jupyter or Colab cell. The `MODE` setting near the top of the file chooses what runs; the comment block at the top of the file explains the protocol in full.

| `MODE` | What runs | Time |
|---|---|---|
| `"calibrate"` | Model-free quantities of the generator on the development seeds, checked against targets and identities written down before the run. No model is trained. Prints the freeze fingerprint. | about a minute |
| `"tune"` | Chooses `max_features` of the random forest on the development seeds. | |
| `"quick"` | Checks that the whole pipeline runs (development seeds, small sizes). Its numbers are not results. | about 10 min |
| `"final"` | The paper's run on the final seeds. | 38 min on 11 cores; about 4 h on 2 cores |

The file is set to `MODE = "final"`. Before it starts, the final run re-checks every target and identity on the development seeds and stops if one fails. Each seed is saved to `results/checkpoints/` as soon as it is done, so an interrupted run resumes when it is started again with the same code and settings. On Colab, point `RESULTS_DIR` to a mounted Google Drive folder, or a recycled runtime loses the checkpoints. The final run also writes the long result tables as CSV files to `results/` (for example `metrics_long.csv`).

## Reproducing the paper's numbers

- **Freeze fingerprint: `80305e8737e47afa`.** The script prints `Freeze fingerprint: 80305e8737e47afa (settings + code)` when it starts, in every mode, and the final run also writes it to `results/freeze_fingerprint.txt`. It is the first 16 hex digits of a SHA-256 of the frozen analysis plan, the targets and identities, the generator and learner settings, and the source of every function in the file. Changing any of them changes the fingerprint. `MODE` and the other settings at the top of the file are not part of it.
- **Seeds:** development seeds 0–9 (calibration, tuning and the quick check only), final seeds 100–119 (all results in the paper), sweep seeds 200–204 (one-at-a-time parameter sweeps). The final and sweep seeds were not run before the freeze.
- **Main settings:** 12 simulated controllers (4 per archetype) × 100 conflicts × 4 contexts = 4,800 decisions per seed; decision noise β = 0.35; random forest with 100 trees, at least 2 samples per leaf and 16 features tried per split; second learner: histogram-based gradient boosting (learning rate 0.1, 200 iterations, 31 leaves, at least 20 samples per leaf).
- **Software of the paper's run:** Python 3.13.5, scikit-learn 1.6.1.

Where the paper's results appear in the printed output of the final run:

| Paper | Printed sections |
|---|---|
| 5.1 Seen controllers and ceilings (Table 2, seen controllers) | 2 (manipulation checks and rules), 3 (recovery) |
| 5.2 Held-out controllers, H1 and H2 (Table 2, held-out controllers) | 4 (the two pre-specified contrasts), 5 (held-out controllers) |
| 5.3 Sample size, second learner and sweeps | 6, 13 |
| 5.4 Safety filter, control worlds, lexicographic policy, placebos | 8, 9, 10 |

## Statistics

For fixed settings every contrast is a constant that more seeds estimate more precisely. The paper therefore reports estimates: means over the 20 final seeds of per-seed differences, with 95% Monte-Carlo intervals. The script prints these, and also the one-sided tests with Holm-adjusted p-values that were specified in the frozen plan. Intervals and p-values measure the Monte-Carlo error of the simulation, not inference about real controllers.

## Contact

Jihane Belhou, Computer Science Research Laboratory, Faculty of Sciences, Ibn Tofail University, Kénitra, Morocco (jihane.belhou@uit.ac.ma)
