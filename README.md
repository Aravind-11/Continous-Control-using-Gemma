# Investigating optimal policy search via LLMS for Reinforcement Learning tasks

## Motivation

The aim of this project is to explore whether an instruction-tuned LLM (Gemma-2B) can be fine-tuned into a decision-making policy for the Frozen Lake environment using only expert Q-values as supervision. The model is not asked to predict Q-values directly; instead, it generates a reasoning trace (`<think>`) and selects a final action (`<answer>`). Expert Q-values are then used to evaluate how good that chosen action is. The motivation is to test whether preference-based reinforcement learning (GRPO) can push a language model to behave like an optimal policy purely from scalar rewards derived from Q-value differences, without any supervised labels or regression losses.

## Method

For each state, the model is prompted with a text version of the grid and asked to produce a chain-of-thought and a final action. After the model selects an action, we compute a reward from the expert Q-values:

1. Let the expert Q-values be $[Q_{\text{left}}, Q_{\text{down}}, Q_{\text{right}}, Q_{\text{up}}]$.
2. Let $\hat{a}$ be the action predicted by the model.
3. We take the expert's Q-value for that action, $Q_{\text{expert}}(\hat{a})$, and convert it into a normalized advantage:

$$
\text{advantage}(\hat{a}) = \frac{Q_{\text{expert}}(\hat{a}) - Q_{\min}}{Q_{\max} - Q_{\min}}
$$

where $Q_{\max}$ and $Q_{\min}$ are the maximum and minimum expert Q-values for that state. This yields a value in $[0,1]$, where 1 corresponds to choosing the optimal action and 0 corresponds to choosing the worst action.

4. A small bonus is added if the model exactly matches the expert's optimal action.
5. A separate format reward checks whether the model uses proper `<think>` and `<answer>` tags.
6. The final reward passed to GRPO is the average of the format reward and the normalized advantage.

This reward is then used by GRPO: multiple sampled completions are compared, higher-reward completions are favored, and LoRA adapter weights are updated so the model becomes more likely to choose high-advantage actions in the future.

## Results

Although the model reliably produced structured outputs, it did not learn a stable or improving policy: the actions chosen by the LLM did not consistently correspond to higher expert Q-values, and training runs did not converge. The root issue is that the model receives only a single scalar advantage-like reward after choosing an action, without any direct supervision of Q-values or demonstration actions. This makes the learning signal sparse, high-variance, and difficult for GRPO to exploit in such a small environment. The results suggest that LLMs require either supervised action labels, direct Q-value regression, or richer reward shaping before preference-based RL can successfully teach grounded control behavior in this setting.

## Alternative Approach: ProPS (What Worked)

While my fine-tuning approach did not converge, my professor and his team were alternatively working on a different idea that proved successful. They used **in-context learning** to predict actions, and their method worked well. The paper was accepted at **NeurIPS 2025**.

The team introduced **Prompted Policy Search (ProPS)** where they added the experiences to the prompts continually without fine tuning the model and the LLM seems to have learned a policy by itself after several iterations. 

The post on the [project website](https://props-llm.github.io/) illustrates the policy search process of ProPS in the various environments.

## The following code illustrates how to fine-tune an LLM using GRPO for RL tasks in OpenAI Gym Environments.

### Running the code

### Requirements

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
