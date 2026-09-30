# Dynamical-Multitask-Prediction

This is a repository containing my work for my Research Comprehensive Exam as part of my PhD program at the University of Toronto.

This project serves as a review and extension of the paper [*Efficient Spatio-Temporal Gaussian Regression via Kalman Filtering*](https://doi.org/10.1016/j.automatica.2020.109032) authored by Marco Todescato, Andrea Carron, Ruggero Carli, and Gianluigi Pillonetto.

This repository includes a transformation of the authors' code from MATLAB into Python, a written review of the authors' paper, and an extension of their work which is geared towards multi-output (vector-valued) responses, as these cases are not discussed in detail in the original paper.

## Notes

This GitHub repository contains an unaltered copy of [the GitHub repository](https://github.com/MarcoTodescato/Efficient-GP-Regression-via-Kalman-Filtering) developed by Todescato et al. for their paper.

The code written by Todescato et al. is stored in the sub-directory [/Code/Todescato-Code/](Code/Todescato-Code/). This copy of their repository is included to ensure full reproducibility, as the datasets used in their paper are directly transformed into Python-readable CSV files (see [/Code/convert-data.py](Code/convert-data.py)) as a part of this project. 

Though the code is available within this repository, it is strongly recommended to visit the authors' original repository for a fuller view of the development of these code files.