# Federated Learning for Text Classification

This repository contains a federated learning implementation for text classification tasks, with a focus on security aspects like backdoor attacks and defenses.

## Project Summary

This project implements federated learning for text classification using transformer-based models (BERT and DistilBERT). It focuses on three main scenarios:

1. **Clean Federated Learning**: Standard FL training on text datasets
2. **Backdoor Attacks**: Implementation of backdoor attacks in federated learning settings
3. **Defense Mechanisms**: Various defense techniques against backdoor attacks

The implementation supports both SST-2 (sentiment classification) and AG News datasets, and includes different parameter tuning approaches, including LoRA (Low-Rank Adaptation).

## Project Structure

- `core/`: Core implementation files
  - `FL_text.py`: Main federated learning implementation
  - `options.py`: Command-line argument definitions
  - `utils.py`: Utility functions
  - `update.py`: Client update algorithms
  - `defense.py` & `defense_utils.py`: Defense mechanisms against attacks
  - `generate_attack_syn_data.py`: Generates synthetic attack data
  - `sampling.py`: Data sampling functionality
  - `gptlm.py`: GPT language model utilities

- `models/`: Model definitions
  - `model.py`: Base model definitions
  - `resnet.py`: ResNet implementation
  - `update_image.py`: Image model update functionality

- `notebooks/`: Jupyter notebooks for experiments
  - `pilot_experiment.ipynb`: Initial experiments
  - `test.ipynb`: Testing notebook
  - `attack_experiment.ipynb`: Experiments with attacks

- `scripts/`: Execution scripts
  - `run_fl_text.sh`: Main execution script

- `data_files/`: Data files including synthetic and attack data
  - Attack and clean synthetic data for SST-2 and AG News
  - Requirements file

- `data/`: Dataset directory

- `results/`: Results of experiments

- `save/` & `save_model/`: Saved model states and checkpoints

- `logs/`: Training logs

- `figures/`: Generated figures and visualizations

## Main Program

The main program is `core/FL_text.py`, which implements federated learning with three main modes:
- Clean federated learning
- Backdoor attacks
- Defense mechanisms

## Usage

To run the program:

```bash
cd scripts
./run_fl_text.sh
```

See the script contents for available configuration options. 