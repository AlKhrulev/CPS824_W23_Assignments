import numpy as np
from numpy import random

# States
#    +---+---+---+---+
#    | 0 | 1 | 2 | 3 |
#    +---+---+---+---+
#    | 4 | 5 | 6 | 7 |
#    +---+---+---+---+
#    | 8 | 9 |10 |11 |
#    +---+---+---+---+
#    |12 |13 |14 |15 |
#    +---+---+---+---+
# 0 & 15 are terminal state

# A = [up, down, left, right]
# Rewards R is -1 on all transitions until the terminal state is reached, 0 for terminal state
R = np.ones((4, 4)) * -1
R[0, 0] = 0
R[3, 3] = 0

# Policy Evaluation
# For each state 
# V[s] = 
# Sum for each action a: Pr(a|s) *
# Sum for each (next state, reward): Pr((next state, reward) |s,a) * [immediate reward + gamma*V[s]]

print(R)