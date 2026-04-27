# EcoCorona-MAP

AI-driven framework for protein-guided control of environmental nano-bio interfaces.

## Overview
This repository contains a minimal reproducible implementation of the EcoCorona-MAP framework proposed in the manuscript. The framework uses a Graph Neural Network (GNN) to predict protein–nanoparticle adsorption and supports peptide-based modulation strategies.

## Contents
- model.py: Simplified GNN model structure
- train.py: Training and evaluation script
- data/sample_data.csv: Sample processed dataset
- requirements.txt: Required Python libraries

## Requirements
- Python 3.10
- PyTorch
- NumPy
- Pandas

Install dependencies using:
pip install -r requirements.txt

## Usage
Run the training script:
python train.py

## Data Source
Proteomic data used in this study are derived from publicly available databases such as UniProt (https://www.uniprot.org/), along with experimentally generated datasets described in the manuscript.

## Note
This repository provides a minimal reproducible workflow for demonstration purposes. Full experimental datasets are available from the corresponding author upon reasonable request.

## Author
Surendar Aravindhan
