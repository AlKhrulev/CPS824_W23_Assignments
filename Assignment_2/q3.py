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

# Rewards R is -1 on all transitions until the terminal state is reached, 0 for terminal state
R = np.ones((16)) * -1
R[0] = 0
R[15] = 0

V = np.zeros((16))

# Actions A = [up, down, left, right]
actionDirection = [(0,-1), (0,1), (-1, 0), (1, 0)]
# Adjacent cells for each action
adjacentOffset = [[2, 3], [2, 3], [0, 1], [0, 1]]

def coordinateToIndex(dir):
    return dir[0] + dir[1] * 4

# returns whether dir a will keep the agent within the board at state s
def withinBounds(s, a):
    x = s % 4 + a[0]
    y = s // 4 + a[1]
    boundsCheck = x >= 0 and x < 4 and y >= 0 and y < 4
    # print("withinBounds", s, (x,y) , boundsCheck)
    return boundsCheck

# Init Policy, equiprobable random policy
Pi = np.ones((16, 4))* 0.25
p1 = 0.8
p2 = 0.1

# Next state probability Pr[s][a][s']
Pr = np.zeros((16, 4, 16))
# Init Next state probability, 0 by default
for s in range(16):
    for a in range(4):
        # Adjacent cells for action
        adjacentPr = (1 - p1 - p2) / 2
        adDir1 = actionDirection[adjacentOffset[a][0]]
        adDir2 = actionDirection[adjacentOffset[a][1]]

        if (withinBounds(s, actionDirection[a])):
            # IntendedNextState within grid
            intendedNextState = s + coordinateToIndex(actionDirection[a])
            Pr[s][a][intendedNextState] = p1
            Pr[s][a][s] = p2

            if(withinBounds(intendedNextState, adDir1)):
                ad1 = intendedNextState + coordinateToIndex(adDir1)
                Pr[s][a][ad1] = adjacentPr
            else:
                # if adjacent not within bounds, "probability of moving in the desired direction is a bit higher" (Q10)
                Pr[s][a][intendedNextState] += adjacentPr

            if(withinBounds(intendedNextState, adDir2)):
                ad2 = intendedNextState + coordinateToIndex(adDir2)
                Pr[s][a][ad2] = adjacentPr
            else:
                # if adjacent not within bounds, "probability of moving in the desired direction is a bit higher" (Q10)
                Pr[s][a][intendedNextState] += adjacentPr
        else:
            # intendedNextState out of grid
            Pr[s][a][s] = p1 + p2

            if(withinBounds(s, adDir1)):
                ad1 = s + coordinateToIndex(adDir1)
                Pr[s][a][ad1] = adjacentPr
            else:
                # if adjacent not within bounds, probability of staying at s higher
                Pr[s][a][s] += adjacentPr

            if(withinBounds(s, adDir2)):
                ad2 = s + coordinateToIndex(adDir2)
                Pr[s][a][ad2] = adjacentPr
            else:
                # if adjacent not within bounds, probability of staying at s higher
                Pr[s][a][s] += adjacentPr
        print(s, actionDirection[a], Pr[s][a])


# # Policy Evaluation
# for i in range(16):
#         # For each state sum expected reward of all possible actions
#         for a in range(4):
#             # For each action sum expected reward of all possible (next state, reward) pairs

# V[s] = 
# Sum for each action a: Pr(a|s) *
# Sum for each (next state, reward): Pr( |s,a) * [immediate reward + gamma*V[s]]
