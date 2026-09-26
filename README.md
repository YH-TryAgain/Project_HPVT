# Project_HPVT

Research code and experimental data for **"HPVT: A Hierarchical Parallel Verification Tree Protocol for Missing Tag Identification in RFID Systems."**

HPVT combines hierarchical prefix-tree pruning, local ALOHA verification, and query aggregation for missing-tag identification in large-scale RFID systems. This repository contains the HPVT simulator, baseline implementations, experiment scripts, saved results, and figures used in the accompanying manuscript.

## Requirements

- Python 3.10 or later
- NumPy
- pandas
- Matplotlib
- tqdm

Install the Python dependencies with:

```bash
python -m pip install numpy pandas matplotlib tqdm
```

## Repository Structure

- `framework.py`: simulation framework and timing model
- `hpvt_algo.py`: HPVT implementation
- `cpt_algo.py`, `iip_algo.py`, `crmti_algo.py`, `ctmti_algo.py`, `ecumi_algo.py`: baseline protocols
- `exp*.py`: experiment entry points
- `results/`: saved experimental data
- `plots/`: figures generated from the experiments

Directories with Chinese names contain preliminary results or archived development versions and are not required to run the main experiments.

## Experiments

| Paper figure | Experiment | Script | Main output |
|---|---|---|---|
| Figure 6 | HPVT threshold sensitivity | `exp1_2.py` | `plots/fig1.2_multimetric_threshold_sensitivity.png` |
| Figure 7 | Performance versus missing rate | `exp2_1.py` | `plots/fig2.1_multimetric_missing_rate_analysis.png` |
| Figure 8 | Scalability versus tag population | `exp2_2.py` | `plots/fig2.2_multimetric_scalability_analysis.png` |
| Figure 9 | Robustness under different ID distributions | `exp2_3.py` | `plots/fig2.3_robustness_analysis_new_dists.png` |

Run an experiment from the repository root:

```bash
python exp1_2.py
python exp2_1.py
python exp2_2.py
python exp2_3.py
```

The scripts use multiprocessing and write generated data and figures to `results/` and `plots/`. Runtime depends on the machine and experiment size. Because the simulations are stochastic, newly generated values may differ from the archived results.

## Citation

If you use this repository, please cite the accompanying manuscript:

```bibtex
@misc{yang2026hpvt,
  author = {Hong Yang and Xiaolin Jia and Yajun Gu and Zhong Du and Hongquan Zhou},
  title  = {HPVT: A Hierarchical Parallel Verification Tree Protocol for Missing Tag Identification in RFID Systems},
  year   = {2026},
  note   = {Manuscript and source code},
  url    = {https://github.com/YH-TryAgain/Project_HPVT}
}
```

Repository: <https://github.com/YH-TryAgain/Project_HPVT>
