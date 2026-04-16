# CLAS_RICH_ML

Machine-learning-based particle identification (PID) workflow for the CLAS12 RICH detector at Jefferson Lab. Contact rmilt003@ucr.edu for questions.

This repository contains the **post-simulation ML pipeline**:
1. Convert simulated `.hipo` files to `.root`
2. Process `.root` files into ML-ready `.h5`
3. Train a GravNet-based model
4. Run inference
5. Make performance/diagnostic plots

> The model implementation is based on the GravNet architecture from Qasim et al. (EPJC 79, 608, 2019), implemented directly in this codebase.

---

## End-to-end workflow

### 1) Generate data (from the `data_generation/` directory in this branch)
Data generation is handled by the files under `data_generation/` in this repository.

The main files in that directory are:
- `lund_generator.cpp`: event generation helper for LUND-style input.
- `*.gcard`: GEMC detector/run cards.
- `*.yaml`: run configuration files for different CLAS12 settings.
- `submit_rich_job.sh`: batch submission helper for generating/simulating jobs.

After generation/reconstruction, you should have CLAS12 output in `.hipo` format.

Some notes on generation:
The script I used to submit jobs is submit_rich_jobs.sh. You'll need to adjust all of the directories in that file so you're not using mine.

That script assumes you already have lund files generated. I generate lund files with lund_generator.cpp. Please change the output directory on [line 98](https://github.com/rymilton/CLAS_RICH_ML/blob/main/data_generation/lund_generator.cpp#L98). You just run this script using `root lund_generator.cpp`. Lastly, you'll need to change [the schema path in the reconstruction yaml file](https://github.com/rymilton/CLAS_RICH_ML/blob/main/data_generation/rga_spring2018.yaml#L76).

---

### 2) Convert `.hipo` to `.root`
Use the converter in this repository:
- `RICH_hipo2root.cxx`
- `hipo2root.sh`

Example compile command for the CLAS12 farm environment:

```bash
g++ RICH_hipo2root.cxx -o RICH_hipo2root \
    -I/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/hipo/4.2.0/include/hipo4 \
    -L/u/scigroup/cvmfs/hallb/clas12/sw/almalinux9-gcc11/local/hipo/4.2.0/lib \
    -lhipo4 `root-config --cflags --libs`
```

Then run:

```bash
bash hipo2root.sh
```

---

### 3) Process `.root` files with `data_processor.py`
`data_processor.py` converts selected ROOT banks/events into model-ready HDF5 files.

What `data_processor.py` does at a high level:
- Opens ROOT files with `uproot` and reads relevant banks (`RICH::Hit`, `REC::Traj`, `MC::Particle`, `REC::Particles`, `RICH::Particle`).
- Applies event/particle quality cuts (trigger selection, valid RICH hits, multiplicity, momentum, charge, sector, trajectory conditions).
- Matches reconstructed particles to MC truth for supervised labels.
- Builds ML features (RICH hit coordinates/times + trajectory/global features), scales them, and writes `.h5` outputs.
- Optionally performs train/test splitting based on `configs/data.yaml`.

Run (single chunk):

```bash
python3 data_processor.py \
  --config data.yaml \
  --config_directory ./configs/
```

Run (subset/chunk):

```bash
python3 data_processor.py \
  --config data.yaml \
  --config_directory ./configs/ \
  --start_file_index 0 \
  --end_file_index 500 \
  --name_suffix 0-500
```

If you need to process a large ROOT sample in batches, use:

```bash
bash data_processor_loop.sh
```

`data_processor_loop.sh` is a convenience wrapper that iterates over the ROOT files in fixed-size chunks and repeatedly calls `data_processor.py` with start/end indices and a suffix per chunk.

---

### 4) Train a model
Train with:

```bash
python3 train.py \
  --config data.yaml \
  --training_config training.yaml \
  --config_directory ./configs/
```

Model checkpoints and training artifacts are saved under `MODEL_SAVE_DIRECTORY` from `configs/training.yaml`.

What `train.py` does:
- Loads the processed training HDF5 files from `SAVE_DIRECTORY`.
- Creates a padded batched representation of variable-length hit lists.
- Builds the GravNet-based network from `model.py`.
- Trains with configured optimizer/scheduler settings.
- Runs validation each epoch, writes loss history, and saves checkpoints (including best model).

---

### 5) Run inference
After training, run:

```bash
python3 inference.py \
  --config data.yaml \
  --training_config training.yaml \
  --config_directory ./configs/
```

This writes `test_predictions.pt` into `MODEL_SAVE_DIRECTORY`.

What `inference.py` does:
- Loads test HDF5 files and the trained checkpoint (`checkpoints/best_model.pth`).
- Runs forward passes to get class probabilities.
- Stores model outputs and supporting metadata (truth labels, reconstructed PID, RICH PID/RQ, momentum, trajectory quantities, hit-level arrays/masks) in `test_predictions.pt`.

---

### 6) Make plots with `analysis.py`
Run:

```bash
python3 analysis.py
```

`analysis.py` reads prediction outputs and produces PID performance plots.
It compares the ML model with conventional reconstructed/RICH PID, computes efficiency-style metrics (including binned studies), and saves publication-style plots to the configured output directory.

---

## Configuration-driven stages (important)
Stages **3–6** are controlled by config files in `configs/`:

- `configs/data.yaml`
  - Input ROOT directory (`DATA_DIRECTORY`)
  - Processed output directory (`SAVE_DIRECTORY`)
  - File naming and event-selection cuts
  - Train/test split behavior

- `configs/training.yaml`
  - Training hyperparameters (batch size, LR, epochs)
  - Model architecture dimensions
  - Scheduler and early-stopping settings
  - Model output directory (`MODEL_SAVE_DIRECTORY`)

If paths or cuts are wrong for your environment, update these config files first.

---

## Repository structure (key files)

- `data_generation/` — simulation/data-generation inputs and job helpers (LUND/gcard/yaml submission setup)
- `RICH_hipo2root.cxx` — C++ converter that reads CLAS12 HIPO events and writes ROOT TTrees/banks for downstream preprocessing
- `hipo2root.sh` — batch helper to run `RICH_hipo2root` over a directory of HIPO files
- `data_processor.py` — ROOT → HDF5 preprocessing and label-building pipeline with cuts/scaling/splitting
- `data_processor_loop.sh` — batch wrapper around `data_processor.py` for chunked processing of large datasets
- `dataset.py` — HDF5 dataset loader used by training/inference code
- `model.py` — GravNet-based architecture implementation used for binary pion/kaon classification
- `train.py` — training entrypoint (data loading, batching, optimization, scheduler, checkpointing)
- `inference.py` — inference entrypoint that runs the trained model and serializes prediction artifacts
- `analysis.py` — plotting/metrics script for model-vs-conventional PID evaluation
- `configs/data.yaml` and `configs/training.yaml` — runtime configuration

---

## Notes

- This code is written assuming CLAS12-style data products/banks.
- Absolute paths in the sample configs are environment-specific; adjust to your machine or farm location.
