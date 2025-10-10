# CLAS_RICH_ML
This repository is for the development of ML-based PID for the RICH at CLAS12 in Jefferson Lab.

The model is based on the GravNet model from  [Qasim, S.R., Kieseler, J., Iiyama, Y. et al. Learning representations of irregular particle-detector geometry with distance-weighted graph networks. Eur. Phys. J. C 79, 608 (2019).](https://link.springer.com/article/10.1140/epjc/s10052-019-7113-9) We have written the model implementation ourselves based on the paper, rather than use the version in [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.nn.conv.GravNetConv.html).

## Dataset
The model is trained using simulated data based on the RGA configuration. The path to the data on the farm is `/work/clas12/pecar/RICH/gemcSim/clas12-rich-sim/clas12Tags-5.10/recoFiles/DIS/default/allOpticsNominal/`. The scripts in this repository require that the .hipo files be converted to .root files. This can be done with [RICH_hipo2root](https://github.com/rymilton/CLAS_RICH_ML/blob/main/RICH_hipo2root.cxx) and [hipo2root.sh](https://github.com/rymilton/CLAS_RICH_ML/blob/main/hipo2root.sh).

## Workflow
The main workflow is as follows:
1. Convert .hipo files to .root files
2. Run training with [train.py](https://github.com/rymilton/CLAS_RICH_ML/blob/main/train.py)
3. Run inference with [inference.py](https://github.com/rymilton/CLAS_RICH_ML/blob/main/inference.py)
4. Run analysis with [analysis.py](https://github.com/rymilton/CLAS_RICH_ML/blob/main/analysis.py)
