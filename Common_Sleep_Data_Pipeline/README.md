# CSDP - Common Sleep Data Pipeline

This repository contains functions and classes for preprocessing and data-loading PSG data, which can then be used for training/predicting with the U-Sleep model.

There are three submodules:
- "csdp_datastore": Preprocessing of PSG datasets. If you already have access to compatible HDF5 files (for example from ERDA), you can ignore this submodule.
- "csdp_pipeline": PyTorch based dataloading of the preprocessed HDF5 files from the datastore submodule.
- "csdp_training": PyTorch Lightning based module of U-Sleep to be used for training, validation, test and simple predictions.

Check out https://gitlab.au.dk/tech_ear-eeg/sleep-code/csdp-demonstration for installation guide and demo scripts