# EM-DPS

Electromagnetic sensitivity transfer for diffusion posterior sampling in GPR full-waveform inversion.

A radar-conditioned diffusion model generates permittivity candidates. Sparse late-step FDTD sensitivities guide bounded sampling updates; final candidates are selected by waveform MSE.

## Installation

Python 3.10+, Linux, an NVIDIA GPU, CUDA-enabled PyTorch, the CUDA toolkit (`nvcc`) and GNU Make are required for inversion.

```bash
pip install -r requirements.txt
make -C fdtd
python scripts/smoke_fdtd.py
```

## Data and training

The full HDF5 dataset and trained checkpoint are external assets (see [docs/DATA.md](docs/DATA.md)). Public download links are not yet available. Bundled results and diagnostic arrays support inspection and figure reproduction.

```bash
python scripts/prepare_config.py --h5 /absolute/path/to/raw.h5
python -m methods.emdps train --cfg cfg.yaml
```

For inference, place the trained checkpoint at `runs/emdps/ckpt/best.pt`.

## Inversion and experiments

Run the following commands from the repository root.

```bash
python scripts/run_inversion.py --method em-dps --cases 128 --out runs/main_em_dps
python scripts/run_inversion.py --method no-dps --cases 128 --out runs/main_no_dps
python -m analysis.compare --dps runs/main_em_dps/dps_best_by_case.csv --nodps runs/main_no_dps/dps_best_by_case.csv --out runs/comparison
python -m experiments.scheduling.run_schedule_ablation --cases 32 --out runs/scheduling
python -m experiments.mechanism.diagnose --out runs/mechanism
python -m experiments.mechanism.save_state_updates
```

Both main methods use 32 steps and four candidates with the same seed and final ranking. EM-DPS applies corrections at steps 16, 20, 24 and 28. The launcher also supports `--experiment noise` and `--experiment mismatch`; see `--help`. Noise affects guidance and evaluation while retaining the original network condition; frequency mismatch affects guidance only.

## Repository

| Path | Contents |
| --- | --- |
| `methods/` | Conditional diffusion, sampling, preprocessing and physical guidance |
| `analysis/` | Method comparison, dispersion, failures and runtime analysis |
| `fdtd/` | Differentiable electromagnetic modelling and CUDA kernels |
| `configs/` | Model and experiment configuration |
| `docs/` | Data access information and external-asset checksums |
| `experiments/` | FWI baselines, robustness, scheduling and update-transfer experiments |
| `figures/` | Plotting scripts and lightweight numerical inputs |
| `results/` | Reported numerical results |
| `scripts/` | Configuration, inference and verification entry points |

Check the bundled main comparison without a GPU or external assets:

```bash
python scripts/check_source_results.py
```

For example, redraw waveform and mechanism panels using:

```bash
python figures/waveforms/plot_compact.py --data figures/waveforms/raw --out plots/waveforms
```
