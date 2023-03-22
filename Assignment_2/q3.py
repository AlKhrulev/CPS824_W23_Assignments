# Alexander Khrulev 500882732
# Mahan Pandey 500881861

import numpy as np
from numpy import random

def coordinateToIndex(dir):
    return dir[0] + dir[1] * 4

# returns whether dir a will keep the agent within the board at state s
def withinBounds(s, a):
    x = s % 4 + a[0]
    y = s // 4 + a[1]
    boundsCheck = x >= 0 and x < 4 and y >= 0 and y < 4
    # print("withinBounds", s, (x,y) , boundsCheck)
    return boundsCheck

def printState(s):
    for y in range(4):
        for x in range(4):
            # print("(%2d" % (x + y*4) + ", %.2f" % s[x + y * 4] + ")", end='')
            print(" % 2.2f|" % s[x + y * 4], end='')
        print()

def printPi(Pi):
    arctionArrow = [" ↑ ", " ↓ ", " ← ", " → "]
    for y in range(4):
        for x in range(4):
            ind = x + y * 4
            optimalAction = np.argmax(Pi[ind])
            if(ind == 0 or ind == 15):
                print(" x |", end='')
            else:
                # print(str(actionDirection[optimalAction]) + "|", end='')
                print(arctionArrow[optimalAction] + "|", end='')
                # print(str(Pi[x + y * 4]) + "|", end='')

        print()

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

# p1 = 0.8
# p2 = 0.1
# gamma = 0.95
# theta = 0.001

p1 = float(input("Input p1: "))
p2 = float(input("Input p2: "))
gamma = float(input("Input gamma: "))
theta = float(input("Input theta: "))

# Rewards R is -1 on all transitions until the terminal state is reached, 0 for terminal state
# R = np.ones((4)) * -1
R = [float(input("r(up): ")), float(input("r(down): ")),float(input("r(left): ")),float(input("r(right): "))]

def getReward(s, a):
    if(s == 0 or s == 15):
        return 0
    else:
        return R[a]

# Actions A = [up, down, left, right]
actionDirection = [(0,-1), (0,1), (-1, 0), (1, 0)]
# Adjacent cells for each action
adjacentOffset = [[2, 3], [2, 3], [0, 1], [0, 1]]

# Init Policy, equiprobable random policy
Pi = np.ones((16, 4))* 0.25

# Next state probability Pr[s][a][s']
Pr = np.zeros((16, 4, 16))
# Init Next state probability, 0 by default
for s in range(16):
    for a in range(4):
        if(s == 0 or s == 15):
            # Terminal state is absorbing
            Pr[s][a][s] = 1
        else:
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
        # print(s, actionDirection[a])
        # printState(Pr[s][a])