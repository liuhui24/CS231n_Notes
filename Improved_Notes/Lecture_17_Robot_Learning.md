# CS231n (Spring 2025) — Lecture 17: Robot Learning

> **Instructor:** Yunzhu Li
> **Date:** May 29, 2025

---

## 1. From Perception to Action

So far in CS231n:

| Paradigm | Data | Goal |
|----------|------|------|
| **Supervised Learning** | $(x, y)$ — data with labels | Learn a mapping $x \rightarrow y$ |
| **Self-Supervised Learning** | $x$ — just data | Learn underlying structure |

**Today — Robot Learning:** An **agent** performs **actions** in an **environment** and receives **rewards**. The goal is to learn how to act to **maximize cumulative reward**.

```
Agent → Action → Environment → Reward → Agent → ...
```

---

## 2. Problem Formulation: Markov Decision Process (MDP)

An MDP is defined by:

| Component | Notation | Description |
|-----------|----------|-------------|
| **State Space** | $\mathcal{S}$ | The set of all possible situations the agent can be in |
| **Action Space** | $\mathcal{A}$ | The set of actions the agent can take |
| **Transition Function** | $P(s_{t+1} \mid s_t, a_t)$ | Probability of next state given current state and action |
| **Reward Function** | $R(s_t, a_t)$ | Immediate reward for taking action in a state |
| **Discount Factor** | $\gamma \in [0, 1]$ | How much to value future rewards vs. immediate rewards |

**Goal:** Find a policy $\pi(a \mid s)$ that maximizes expected cumulative discounted reward:

$$\mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$

### MDP Examples Across Domains

| Domain | State | Action | Reward |
|--------|-------|--------|--------|
| **Cart-Pole** | Angle, angular speed, position, velocity | Horizontal force on cart | +1 each timestep pole is upright |
| **Robot Locomotion** | Joint angles, positions, velocities | Joint torques | Upright + forward movement |
| **Atari Games** | Raw pixels of game screen | Joystick directions + button | Score increase/decrease |
| **Go** | Position of all pieces | Where to place next piece | +1 if win, 0 if lose (terminal) |
| **Text Generation** | Current words in sentence | Next word | +1 if correct, 0 otherwise |
| **Chatbot** | Current conversation | Next sentence | Human evaluation |
| **Cloth Folding** | Robot + cloth state | End-effector motions | +1 if cloth is folded |

This generality is powerful: MDPs unify problems across robotics, games, NLP, and more.

---

## 3. Robot Perception

### What Makes Robot Perception Hard

- **Incomplete knowledge** of objects and scenes (occlusion, novel objects)
- **Imperfect actions** that may fail or have unintended effects
- **Environment dynamics** — the world changes, other agents move
- **Multimodal sensing:** Visual (RGB-D), tactile (touch sensors), auditory, proprioception (joint encoders)

### Robot Vision vs. Computer Vision

| | Standard Computer Vision | Robot Vision |
|------|--------------------------|--------------|
| **Embodied?** | No — passive data consumer | Yes — physical body, direct world interaction |
| **Active?** | No — processes given images | Yes — chooses *what* to perceive, *when*, and *how* |
| **Situated?** | Works with abstract data | Deals with the "here and now" of the physical world |

### The Perception-Action Loop

The key challenge: **closing the loop** between perception and action.

- Perception informs action (where is the object? → how to grasp)
- Action changes perception (grasping the object reveals its back side)
- This coupling is fundamental to embodied intelligence

---

## 4. Reinforcement Learning (RL)

RL trains agents that interact with an environment and learn to maximize reward through **trial and error**.

### RL vs. Supervised Learning

| | Supervised Learning | Reinforcement Learning |
|------|---------------------|-----------------------|
| **Training signal** | Ground-truth labels $(x_i, y_i)$ | Scalar rewards from environment |
| **Feedback type** | "This is the correct answer" | "That was good/bad" (evaluative, not instructive) |
| **Data distribution** | Fixed dataset | Depends on agent's own actions |
| **Credit assignment** | Direct (gradient through network) | Delayed — which action caused the reward 100 steps later? |

### Core Approaches

| Approach | Key Idea |
|----------|----------|
| **Value-Based (Q-Learning, DQN)** | Learn $Q(s, a)$ — the expected future reward of taking action $a$ in state $s$; policy = choose action with highest Q-value |
| **Policy Gradient (REINFORCE, PPO)** | Directly optimize the policy $\pi_\theta(a \mid s)$ by gradient ascent on expected reward |
| **Actor-Critic** | Combine both: Actor (policy) decides actions; Critic (value function) evaluates them |

### The Exploration-Exploitation Tradeoff

- **Exploitation:** Do what you know works (maximize immediate reward)
- **Exploration:** Try new actions to discover potentially better strategies
- Balancing these is a central challenge of RL

---

## 5. Model Learning & Model-Based Planning

### Model-Free vs. Model-Based

| | Model-Free RL | Model-Based RL |
|------|--------------|----------------|
| **What it learns** | Policy or value function directly | A model of the environment's dynamics |
| **How it plans** | Implicitly (through trial-and-error) | Explicitly: simulate future states with the learned model, then plan optimal actions |
| **Sample efficiency** | Low — needs many environment interactions | Higher — can "imagine" outcomes without real execution |
| **Examples** | DQN, PPO, SAC | Dyna-Q, MuZero, Dreamer |

### Why Models Matter for Robotics

Real robots are slow, expensive, and can break. Learning a **dynamics model** allows:
- Planning actions in simulation before execution
- Data-efficient learning from limited real-world interactions
- Transfer between related tasks

---

## 6. Imitation Learning

When reward functions are hard to specify or exploration is dangerous:

- **Behavioral Cloning:** Treat expert demonstrations as supervised data — learn $\pi(s) \rightarrow a$ directly
- **Inverse RL:** Infer the reward function that the expert appears to be optimizing
- **DAgger (Dataset Aggregation):** Iteratively collect more data by rolling out the learned policy and asking the expert to correct mistakes

**Challenge:** Distribution shift — the policy encounters states not in the demonstration data, leading to compounding errors.

---

## 7. Robotic Foundation Models & Remaining Challenges

### Emerging Trends

- **Vision-Language-Action models:** Extend VLMs to predict robot actions, enabling instruction-following (e.g., "pick up the red block")
- **Large-scale robot data:** Multi-institution datasets pooling robot experience
- **Sim-to-real transfer:** Train in simulation, deploy on real hardware (domain randomization, system identification)

### Remaining Challenges

| Challenge | Description |
|-----------|-------------|
| **Sample Efficiency** | Real robot data is expensive and slow to collect |
| **Generalization** | What works in one kitchen doesn't work in another |
| **Safety** | Exploration can cause damage |
| **Long-Horizon Tasks** | "Clean the kitchen" requires dozens of sequential steps |
| **Perception Robustness** | Lighting, clutter, novel objects |

---

## Core Takeaways

1. **MDPs** provide a unified framework for sequential decision-making across robotics, gaming, and language.
2. **Robot vision is embodied, active, and situated** — fundamentally different from static image understanding.
3. **Reinforcement learning** learns from evaluative feedback (rewards), not instructive feedback (labels) — the exploration-exploitation tradeoff is central.
4. **Model-based RL** learns environment dynamics for more sample-efficient planning.
5. The field is moving toward **robotic foundation models** that combine vision, language, and action in a unified framework.

---

*Notes compiled from CS231n Spring 2025 Lecture 17 slides and course materials.*
