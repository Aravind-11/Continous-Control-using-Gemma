# Continuous Control using Gemma

This repository contains implementation for training Gemma 2B Instruct model on a continuous control task (Pendulum environment) using Reinforcement Learning from Preference Optimization (GRPO).

## Overview

This project demonstrates how to fine-tune the Google Gemma 2B Instruct model to learn continuous control policies for the Frozen Lake environment using reinforcement learning techniques. The implementation utilizes GRPO (Generative Reinforcement Learning from Preference Optimization) combined with expert demonstrations to train a language model to provide optimal action recommendations.


## Requirements

- Python 3.8+
- PyTorch
- Transformers
- TRL (Transformer Reinforcement Learning)
- PEFT (Parameter-Efficient Fine-Tuning)
- Pandas
- NumPy
- TensorBoard (for monitoring)

## Installation

```bash
# Create and activate a conda environment
conda create -n gemma-rl python=3.8
conda activate gemma-rl

# Install dependencies
pip install torch transformers trl peft datasets numpy pandas
pip install tensorboard
```

## Dataset

The training relies on a dataset of expert demonstrations for the Frozen Lake environment, stored in `expert_demos_batched_avg_q.csv`. This file should contain:
- `state_str`: String representation of the Frozen Lake state
- `action`: Expert action (0: LEFT, 1: DOWN, 2: RIGHT, 3: UP)
- `q_value_left`, `q_value_down`, `q_value_right`, `q_value_up`: Q-values for each possible action

## Training

To train the model, run:

```bash
python train_script.py
```

Alternatively, if using a SLURM environment, use the provided batch script:

```bash
sbatch train_script.sh
```

## Training Configuration

The training configuration uses:
- Gemma 2B Instruct model
- LoRA fine-tuning (rank 16)
- BFloat16 precision
- Batch size of 4 with gradient accumulation steps of 4
- Learning rate of 3e-5 with cosine scheduler
- Combined reward function based on formatting and Q-value optimization

## Monitoring

Training progress can be monitored with TensorBoard:

```bash
tensorboard --logdir ./frozen-lake-gemma-2b-it-grpo
```


## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:
```
@software{continuous_control_gemma,
  author = {Aravind-11},
  title = {Continuous-Control-using-Gemma},
  year = {2025},
  url = {https://github.com/Aravind-11/Continuous-Control-using-Gemma}
}
```

## Acknowledgements

- Google for the Gemma model
- Hugging Face for the Transformers library
- The TRL and PEFT libraries for reinforcement learning with transformers
