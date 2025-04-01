import os
import random
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model

# Set PyTorch memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set Hugging Face cache directory to /scratch/athanina
os.environ["HF_HOME"] = "/scratch/athanina/huggingface_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# --- XML Parser ---
class XMLParser:
    """Parser for extracting content from XML tags."""
    def __init__(self, fields=None):
        self.fields = fields or []

    def parse(self, text):
        """Parse text to extract content within specified XML tags."""
        result = type('obj', (object,), {field: None for field in self.fields})() # Create instance
        for field in self.fields:
            pattern = f"<{field}>(.*?)</{field}>"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                setattr(result, field, match.group(1).strip())
        return result

# --- Prompt Formatting ---
def format_prompt(state_str, system_prompt=None, few_shot=None):
    """Format the prompt for the Frozen Lake task with few-shot examples."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if few_shot:
        messages.extend(few_shot)

    user_content = f"""Current Frozen Lake state:
{state_str}

What action should I take to maximize my chances of reaching the goal safely?
Respond with one of: LEFT, DOWN, RIGHT, UP.

First think through your reasoning in <think> tags, then provide your final answer in <answer> tags.
"""
    messages.append({"role": "user", "content": user_content})
    # Gemma instruction format uses specific tokens
    # Format: "<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
    prompt_str = "<start_of_turn>user\n"
    if system_prompt:
         prompt_str += system_prompt + "\n\n" # Add system prompt if present
    if few_shot:
        for i in range(0, len(few_shot), 2):
            prompt_str += few_shot[i]['content'] # User part
            prompt_str += "<end_of_turn>\n<start_of_turn>model\n"
            prompt_str += few_shot[i+1]['content'] # Assistant part
            prompt_str += "<end_of_turn>\n<start_of_turn>user\n" # Prepare for next user turn

    prompt_str += user_content # Add the final user query
    prompt_str += "<end_of_turn>\n<start_of_turn>model\n" # Signal model to respond

    # Note: The above formatting assumes the tokenizer adds BOS/EOS appropriately.
    # Depending on the specific Gemma tokenizer usage, slight adjustments might be needed.
    # Let's simplify for now and assume the tokenizer handles the structure.
    # Reverting to simpler format for clarity, check Gemma docs if issues arise.
    messages_formatted = []
    if system_prompt:
         messages_formatted.append(f"System: {system_prompt}")
    if few_shot:
        for msg in few_shot:
            messages_formatted.append(f"{msg['role']}: {msg['content']}")
    messages_formatted.append(f"user: {user_content}")
    messages_formatted.append("model:") # Prompt model to start generating

    # Using a simple join for now, ensure tokenizer handles roles if needed
    return "\n".join(messages_formatted)


# --- Frozen Lake Environment ---
class FrozenLakeEnv:
    def __init__(
        self,
        system_prompt="You are an expert at solving the Frozen Lake environment. Analyze the state and provide your reasoning before giving the action.",
        max_steps=1, # Unused currently
    ):
        """Initialize the Frozen Lake environment for RL training."""
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.parser = XMLParser(fields=["think", "answer"])
        # Reduced few-shot examples
        self.few_shot = [
             {
                "role": "user",
                "content": """Current Frozen Lake state:
S F F F
F H F H
F F F H
H F F G

What action should I take to maximize my chances of reaching the goal safely?
Respond with one of: LEFT, DOWN, RIGHT, UP.

First think through your reasoning in <think> tags, then provide your final answer in <answer> tags."""
            },
            {
                "role": "assistant",
                "content": """<think>The goal (G) is at the bottom-right (3,3). From the start (S) at (0,0), moving RIGHT to (0,1) 'F' keeps me on safe ice and progresses toward the goal.</think>
<answer>RIGHT</answer>"""
            },
            {
                "role": "user",
                "content": """Current Frozen Lake state:
F F S F
F H F H
F F F H
H F F G

What action should I take to maximize my chances of reaching the goal safely?
Respond with one of: LEFT, DOWN, RIGHT, UP.

First think through your reasoning in <think> tags, then provide your final answer in <answer> tags."""
            },
            {
                "role": "assistant",
                "content": """<think>I'm at (0,2), and the goal is at (3,3). Moving DOWN to (1,2) 'F' is safe and brings me closer to the goal vertically.</think>
<answer>DOWN</answer>"""
            },
             { # Adding one more distinct example
                "role": "user",
                "content": """Current Frozen Lake state:
F F F F
F H F H
F F S H
H F F G

What action should I take to maximize my chances of reaching the goal safely?
Respond with one of: LEFT, DOWN, RIGHT, UP.

First think through your reasoning in <think> tags, then provide your final answer in <answer> tags."""
            },
            {
                "role": "assistant",
                "content": """<think>I'm at (2,2), and the goal is at (3,3). Moving DOWN to (3,2) 'F' is safe, then I can move RIGHT to (3,3) 'G' to reach the goal.</think>
<answer>DOWN</answer>"""
            }
        ]
        self.dataset = self._load_dataset()
        self.action_map = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}


    def _load_dataset(self):
        """Load and validate expert demonstrations from CSV."""
        try:
            demos_df = pd.read_csv("expert_demos_batched_avg_q.csv")
            required_cols = ["state_str", "action", "q_value_left", "q_value_down", "q_value_right", "q_value_up"]
            if not all(col in demos_df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {set(required_cols) - set(demos_df.columns)}")
            demos_df = demos_df.dropna(subset=required_cols[1:]) # Drop rows with missing Q-values or action
             # Ensure action is integer if it's not already
            if demos_df['action'].dtype != np.int64 and demos_df['action'].dtype != np.int32:
                 # Assuming action might be stored as string name, map it
                 action_map_load = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}
                 # Handle potential errors during mapping
                 demos_df['action'] = demos_df['action'].apply(lambda x: action_map_load.get(str(x).upper(), -1))
                 demos_df = demos_df[demos_df['action'] != -1] # Remove rows with invalid actions
            demos_df['state_str'] = demos_df['state_str'].str.replace('|', '\n', regex=False)

        except FileNotFoundError:
             raise RuntimeError("Failed to load dataset: expert_demos_batched_avg_q.csv not found.")
        except Exception as e:
            raise RuntimeError(f"Failed to load or process dataset: {e}")

        examples = []
        for _, row in demos_df.iterrows():
            prompt = format_prompt(
                row["state_str"],
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
            # Ensure Q-values are floats
            q_values = [
                float(row["q_value_left"]),
                float(row["q_value_down"]),
                float(row["q_value_right"]),
                float(row["q_value_up"])
            ]
            examples.append({
                "prompt": prompt,
                "action": int(row["action"]), # Ensure action is integer index
                "q_values": q_values
            })
        return pd.DataFrame(examples)

    def get_dataset(self, n=None):
        """Return a subset or full dataset."""
        if n and len(self.dataset) > n:
            # Use random_state for reproducibility if needed
            return self.dataset.sample(n=n, random_state=42)
        return self.dataset

    def evaluate_action(self, completion, q_values, expert_action_index):
        """Calculate reward based on Q-values and formatting."""
        parsed = self.parser.parse(completion)
        think_tag = bool(re.search(r"<think>(.*?)</think>", completion, re.DOTALL))
        answer_tag = bool(re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL))
        format_reward = 1.0 if (think_tag and answer_tag) else 0.0

        correctness_reward = 0.0 # Default to 0
        if parsed.answer is not None:
            pred_action_str = parsed.answer.strip().upper()
            pred_action_index = self.action_map.get(pred_action_str, -1)

            if pred_action_index != -1:
                # Ensure Q-values are valid numbers
                if not all(isinstance(q, (int, float)) for q in q_values):
                     print(f"Warning: Invalid Q-values encountered: {q_values}")
                     # Handle invalid Q-values, e.g., return minimal reward or skip
                     return format_reward * 0.1 # Penalize heavily but keep format signal

                expert_q_value = q_values[expert_action_index]
                pred_q_value = q_values[pred_action_index]
                q_max, q_min = max(q_values), min(q_values)

                if q_max > q_min:
                    # Normalize predicted Q-value
                    normalized_pred_q = (pred_q_value - q_min) / (q_max - q_min)
                    # Add bonus for exact match with expert action index
                    exact_match_bonus = 0.2 if pred_action_index == expert_action_index else 0.0
                    # Combine normalized Q + bonus, capped at 1.0
                    correctness_reward = min(1.0, normalized_pred_q + exact_match_bonus)
                else:
                    # Handle case where all Q-values are the same
                    correctness_reward = 1.0 if pred_action_index == expert_action_index else 0.5
            # else: pred_action_index == -1 (invalid action string), correctness_reward remains 0.0

        # Combine format and correctness rewards (equal weighting)
        final_reward = (format_reward + correctness_reward) / 2.0
        return final_reward


    # Updated rubric as suggested
    def get_rubric(self):
        """Return the primary reward function for GRPO training."""
        def combined_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
            # Ensure necessary kwargs are present
            if "q_values" not in kwargs or "action" not in kwargs:
                 raise ValueError("Missing 'q_values' or 'action' in reward function kwargs")

            q_values_list = kwargs["q_values"]
            expert_actions_indices = kwargs["action"] # Assuming 'action' is the expert index
            rewards = []

            if not (len(completions) == len(q_values_list) == len(expert_actions_indices)):
                 print(f"Warning: Mismatch in lengths - completions: {len(completions)}, q_values: {len(q_values_list)}, actions: {len(expert_actions_indices)}")
                 # Handle mismatch, e.g., return default rewards or skip batch
                 return [0.0] * len(completions) # Example: return 0 reward for all

            for i, (completion, q_values, expert_action_index) in enumerate(zip(completions, q_values_list, expert_actions_indices)):
                # Ensure expert_action_index is valid
                if not isinstance(expert_action_index, int) or not (0 <= expert_action_index < 4):
                     print(f"Warning: Invalid expert action index {expert_action_index} at index {i}. Skipping reward calculation.")
                     rewards.append(0.0) # Assign zero reward for invalid data
                     continue

                # Ensure q_values is a list of numbers
                if not isinstance(q_values, list) or len(q_values) != 4:
                     print(f"Warning: Invalid q_values format {q_values} at index {i}. Skipping reward calculation.")
                     rewards.append(0.0)
                     continue

                reward = self.evaluate_action(completion, q_values, expert_action_index)
                # Optional: Add more verbose logging for debugging low rewards
                # if reward < 0.1:
                #    print(f"Debug: Low reward ({reward:.2f}) for completion {i}: {completion[:100]}... | Expert Action Index: {expert_action_index} | Q-values: {q_values}")
                rewards.append(reward)
            return rewards

        # Return only the single combined reward function in a list
        return [combined_reward_func]


# --- Custom Logging Callback ---
class LoggingCallback(TrainerCallback):
    """Callback to log training progress and sample completions."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Basic console logging of metrics reported by Trainer
            print(f"Step {state.global_step}: {logs}")

    # Optional: Log sample completions periodically (can be verbose)
    # def on_step_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
    #     if state.global_step % 50 == 0 and state.global_step > 0: # Log every 50 steps
    #         if 'eval_dataloader' in kwargs: # Check if eval dataloader is available
    #             try:
    #                 # Get a sample from the eval dataset
    #                 eval_loader = kwargs['eval_dataloader']
    #                 batch = next(iter(eval_loader))
    #                 prompt_text = tokenizer.decode(batch['prompt_input_ids'][0], skip_special_tokens=True)
    #                 # Generate a completion
    #                 inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    #                 # Adjust generation parameters as needed
    #                 outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    #                 completion_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #                 print(f"\n--- Sample Completion at Step {state.global_step} ---")
    #                 print(f"Prompt:\n{prompt_text}")
    #                 print(f"Completion:\n{completion_text}\n--------------------------------\n")
    #             except Exception as e:
    #                 print(f"Error generating sample completion at step {state.global_step}: {e}")


# --- Training Function ---
def train_gemma_on_frozen_lake_with_grpo():
    """Train Gemma on Frozen Lake using GRPO with expert Q-value demonstrations."""
    print("Initializing environment and loading dataset...")
    env = FrozenLakeEnv()
    # Use a reasonable subset for training and validation
    train_dataset = env.get_dataset(n=1000) # Increased dataset size slightly
    eval_dataset = env.get_dataset(n=200)
    reward_funcs = env.get_rubric()

    print(f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples.")

    # Load Gemma model and tokenizer
    model_name = "google/gemma-2b-it" # Changed to Gemma 2B Instruct
    print(f"Loading model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left", # Important for generation
        truncation=True,
        max_length=768 # Increased max_length slightly from 512
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, # Use bfloat16 on A100
        device_map="auto"
    )

   
    print("Model and tokenizer loaded.")

    # Configure LoRA
    lora_config = LoraConfig(
        r=16, # Increased rank slightly as Gemma 2B is smaller
        lora_alpha=32, # Standard practice: alpha = 2*r
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # Verify trainable parameters

    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Important: Also update model config if needed, though PEFT/Trainer usually handle this
        model.config.pad_token_id = tokenizer.pad_token_id


    print("Converting pandas DataFrames to Hugging Face Datasets...")
    train_dataset_hf = Dataset.from_pandas(train_dataset)
    eval_dataset_hf = Dataset.from_pandas(eval_dataset)
    print("Dataset conversion complete.")

    # GRPO configuration
    output_dir = "./frozen-lake-gemma-2b-it-grpo"
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=5, # Reduced epochs slightly, instruct model might learn faster
        per_device_train_batch_size=4,  # Increased batch size due to smaller model
        per_device_eval_batch_size=4,   # Increased batch size
        gradient_accumulation_steps=4,  # Adjusted grad accum (effective batch size 16)
        learning_rate=3e-5, # Slightly increased LR for smaller model/LoRA
        weight_decay=0.01,
        logging_steps=10, # Log more frequently
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        bf16=True,       # Use BF16 precision on A100
        fp16=False,      # Disable FP16
        report_to="tensorboard", # <--- Enable TensorBoard logging
        num_generations=4,  # Increased generations back
        beta=0.1, # Adjusted beta slightly
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        remove_unused_columns=False, # Important: Keep 'q_values' and 'action' for reward func
        ddp_find_unused_parameters=False,
    )

    # Initialize GRPO trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset_hf,
    eval_dataset=eval_dataset_hf,
    # tokenizer=tokenizer, # <--- REMOVE THIS LINE
    reward_funcs=reward_funcs,
    callbacks=[LoggingCallback()] # Keep console logging
)
    print("\n" + "="*40)
    print("      Starting GRPO Training with Gemma 2B      ")
    print("="*40 + "\n")
    print("To monitor training with TensorBoard:")
    print(f"1. Ensure 'tensorboard' is installed (`pip install tensorboard`)")
    print(f"2. In a SEPARATE terminal, navigate to the directory containing '{output_dir}'")
    print(f"3. Run: tensorboard --logdir {output_dir}")
    print(f"4. Open the URL provided (usually http://localhost:6006) in your browser.")
    print(f"   (If running on a remote server, you might need SSH port forwarding: ssh -N -L 6006:localhost:6006 user@server)")
    print("\nTraining logs will also appear below...\n")


    try:
        trainer.train()
        success = True
    except Exception as e:
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"            TRAINING ERROR            ")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        success = False

    # --- Saving ---
    final_save_dir = os.path.join(output_dir, "final_model")
    checkpoint_save_dir = os.path.join(output_dir, "checkpoint_on_error")

    if success:
        print("\nTraining completed successfully.")
        try:
            print(f"Saving final model and tokenizer to {final_save_dir}...")
            trainer.save_model(final_save_dir)
            tokenizer.save_pretrained(final_save_dir)
            print("Final model saved.")
        except Exception as e:
            print(f"Error saving final model: {e}")
    else:
        print("\nTraining failed or was interrupted.")
        try:
            print(f"Attempting to save checkpoint to {checkpoint_save_dir}...")
            trainer.save_model(checkpoint_save_dir)
            tokenizer.save_pretrained(checkpoint_save_dir)
            print("Checkpoint saved.")
        except Exception as e:
            print(f"Could not save checkpoint: {e}")

if __name__ == "__main__":
    # Ensure the dataset file exists before starting
    if not os.path.exists("expert_demos_batched_avg_q.csv"):
        print("ERROR: Dataset file 'expert_demos_batched_avg_q.csv' not found.")
        print("Please ensure the dataset is in the same directory as the script.")
    else:
        train_gemma_on_frozen_lake_with_grpo()