# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

## POLICY EVALUATION FUNCTION
![267727933-ee9c6dcf-b579-4b1c-9663-47c8b17a08b4](https://github.com/Lakshmipriya-P-AI/rl-policy-evaluation/assets/93427923/1d10b711-76f7-4ec9-abb5-7222a6890989)

### States
The environment has 7 states:

  * Two Terminal States: G: The goal state & H: A hole state.
  * Five Transition states / Non-terminal States including S: The starting state.

### Actions
The agent can take two actions:

   * R: Move right.
   * L: Move left.
### Transition Probabilities
The transition probabilities for each action are as follows:

   * 50% chance that the agent moves in the intended direction.
33.33% chance that the agent stays in its current state.
16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.




## OUTPUT:
Mention the first and second policies along with its state value function and compare them

## RESULT:

Write your result here
