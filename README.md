# TopoDiff, a guided diffusion model for Topology Optimization

This repository contains the codebase for [Diffusion Models Beat GANs on Topology Optimization](https://arxiv.org/abs/2208.09591).

## Contents

This directory contains:
- A setup file, that installs all the libraries needed and the topodiff package (see the section *Instructions*);
- A directory called *topodiff*, which contains the raw code;
- A directory called *scripts*, which contains the main scripts useful to train all three models and to sample from TopoDiff, using the diffusion model and the two surrogate models;
- An empty directory called *data*, that should be filled with the three datasets (*dataset_1_diff*, *dataset_2_reg* and *dataset_3_class*) available for download [here](https://decode.mit.edu/projects/topodiff/);
- An empty directory called *checkpoints*, that may be filled with the three checkpoints available for download [here](https://decode.mit.edu/projects/topodiff/);
- A directory called *fem_files*, which will contain temporary files used by the FEA solver;
- Five notebooks to help perform easily the training of all three models, the sampling from TopoDiff, and the analysis of the results.

## Getting Started Instructions

To setup the infrastructure needed to run the code, please run the following command at the root of this directory.
```
pip install -e .
```
This should install all the libraries needed to run the code and the topodiff package.

Also make sure you have filled the *data* directory with the three datasets ; and the *checkpoints* directory with the three checkpoints if you do not conduct your own training.

Then, open with Jupyter the notebooks to launch training, sample or analysis.

The order of the trainings does *not* matter and all three trainings are independant.

## More details and contact information

Should you need any additional detail, please check out [this page](https://decode.mit.edu/projects/topodiff/) that gives an overview of the project.

Should you need help with running the code, please contact fmaze@mit.edu or faez@mit.edu.