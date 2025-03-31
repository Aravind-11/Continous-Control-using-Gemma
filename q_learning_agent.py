import os
import random
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import pickle

# Q-Learning implementation for generating expert demonstrations
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, 
                 exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initialize a Q-Learning agent for the Frozen Lake environment.
        
        Args:
            env: Gymnasium Frozen Lake environment
            learning_rate: Alpha parameter for Q-learning
            discount_factor: Gamma parameter for Q-learning
            exploration_rate: Initial epsilon for epsilon-greedy policy
            exploration_decay: Rate at which exploration decreases
            min_exploration_rate: Minimum exploration rate
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table with zeros
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.q_table = np.zeros((self.num_states, self.num_actions))
    
    def choose_action(self, state, deterministic=False):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            deterministic: If True, always choose the best action
            
        Returns:
            action: Selected action
        """
        if not deterministic and random.uniform(0, 1) < self.exploration_rate:
            # Explore: choose a random action
            return self.env.action_space.sample()
        else:
            # Exploit: choose the best action from Q-table
            return np.argmax(self.q_table[state, :])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using the Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Q-learning update formula
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + (1 - done) * self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
        
        # Decay exploration rate
        if done:
            self.exploration_rate = max(
                self.min_exploration_rate, 
                self.exploration_rate * self.exploration_decay
            )
    
    def train(self, num_episodes=5000, evaluation_interval=1000, evaluation_episodes=100, success_threshold=0.7):
        """
        Train the agent using Q-learning and periodically evaluate expertise.
        
        Args:
            num_episodes: Number of episodes to train
            evaluation_interval: How often to evaluate the agent
            evaluation_episodes: Number of episodes for evaluation
            success_threshold: Minimum success rate required to be considered an expert
            
        Returns:
            rewards: List of rewards for each episode
            is_expert: Boolean indicating if agent meets expertise threshold
        """
        rewards = []
        
        for episode in tqdm(range(num_episodes), desc="Training Q-Learning Agent"):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                self.update(state, action, reward, next_state, done)
                state = next_state
            
            rewards.append(total_reward)
            
            # Evaluate periodically
            if (episode + 1) % evaluation_interval == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {episode+1}, Average Reward (last 100): {avg_reward:.2f}, Exploration Rate: {self.exploration_rate:.4f}")
                
                # Check if agent is an expert
                success_rate = self.evaluate(evaluation_episodes)
                print(f"Evaluation: Success rate = {success_rate:.2f} (threshold: {success_threshold})")
                
                if success_rate >= success_threshold:
                    print(f"Agent achieved expert performance at episode {episode+1}")
                    return rewards, True
        
        # Final evaluation after training
        success_rate = self.evaluate(evaluation_episodes)
        print(f"Final Evaluation: Success rate = {success_rate:.2f} (threshold: {success_threshold})")
        is_expert = success_rate >= success_threshold
        
        return rewards, is_expert
    
    def evaluate(self, num_episodes=100):
        """
        Evaluate the agent's performance.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            success_rate: Fraction of episodes where the agent reached the goal
        """
        success_count = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state, deterministic=True)  # Use greedy policy
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state
                
                # Check if episode was successful (reached goal)
                if terminated and reward > 0:
                    success_count += 1
                    break
        
        return success_count / num_episodes
    
    def get_optimal_action(self, state):
        """Get the optimal action for a given state."""
        return np.argmax(self.q_table[state, :])
    
    def get_q_values(self, state):
        """Get Q-values for all actions in a given state."""
        return self.q_table[state, :]
    
    def save(self, filepath):
        """Save the Q-table to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filepath):
        """Load the Q-table from a file."""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
