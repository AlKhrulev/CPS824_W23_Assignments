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
    for y in range(4):
        for x in range(4):
            optimalAction = np.argmax(Pi[x + y * 4])
            print(str(actionDirection[optimalAction]) + "|", end='')
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

p1 = 0.8
p2 = 0.1
gamma = 0.95
theta = 0.001

# Rewards R is -1 on all transitions until the terminal state is reached, 0 for terminal state
R = np.ones((16)) * -1
R[0] = 0
R[15] = 0

# Init V(s) with 0
V = np.zeros((16))

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

# Vnexts = np.zeros(16)
# epoch = 0
# # Policy iteration
# while(True):
#     epoch += 1
#     # Policy Evaluation
#     while(True):
#         delta = 0
#         for i in range(16):
#             lastV = V[i]
#             # For each state sum expected reward of all possible actions
#             nextV = 0
#             for a in range(4):
#                 actionReward = 0
#                 # For each action sum expected reward of all possible (next state, reward) pairs
#                 for j in range(16):
#                     actionReward += Pr[i][a][j] * (R[j] + gamma * V[j])
#                 nextV += Pi[i][a] * actionReward
#             Vnexts[i] = nextV
#             delta = max(delta, abs(nextV - lastV))
        
#         V = np.copy(Vnexts)
#         if(delta < theta):
#             break

#     print("Policy evaluation", epoch)
#     printState(V)

#     print("Old Pi", epoch)
#     printPi(Pi)

#     # Policy Improvement
#     policyStable = True
#     for i in range(16):
#         oldMaxAction = np.argmax(Pi[i])
#         maxAction = 0
#         maxReward = float('-inf')
#         # For each state find action that maximizes expected reward
#         for a in range(4):
#             actionReward = 0
#             # For each action sum expected reward of all possible (next state, reward) pairs
#             for j in range(16):
#                 actionReward += Pr[i][a][j] * (R[j] + gamma * V[j])
#             if actionReward > maxReward:
#                 maxReward = actionReward
#                 maxAction = a

#         if maxAction != oldMaxAction:
#             policyStable = False

#         newPi = np.zeros((4))
#         optimalActionIncrease = 0
#         # Decrease non optimal actions by half
#         for a in range(4):
#             if(a != maxAction):
#                 newPi[a] = Pi[i][a] / 2
#                 optimalActionIncrease +=  Pi[i][a] / 2
#         # Increase optimal action 
#         newPi[maxAction] = Pi[i][maxAction] + optimalActionIncrease
#         # print(np.sum(Pi[i]), Pi[i], newPi)
#         Pi[i] = newPi

#     print("new Pi", epoch, policyStable)
#     printPi(Pi)
#     if policyStable:
#         break




# Value iteration
# Instead of waiting for policy evaluation to converge 
# Find action that maximizes reward in next step
# Update policy evaluation based on best max action
Vnexts = np.zeros(16)
epoch = 0
while(True):
    epoch += 1
    delta = 0
    for i in range(16):
        oldMaxAction = np.argmax(Pi[i])
        maxAction = 0
        maxReward = float('-inf')
        # For each state find action that maximizes expected reward
        for a in range(4):
            actionReward = 0
            # For each action sum expected reward of all possible (next state, reward) pairs
            for j in range(16):
                actionReward += Pr[i][a][j] * (R[j] + gamma * V[j])
            if actionReward > maxReward:
                maxReward = actionReward
                maxAction = a

        if maxAction != oldMaxAction:
            policyStable = False

        newPi = np.zeros((4))
        optimalActionIncrease = 0
        # Decrease non optimal actions by half
        for a in range(4):
            if(a != maxAction):
                newPi[a] = Pi[i][a] / 2
                optimalActionIncrease +=  Pi[i][a] / 2
        # Increase optimal action 
        newPi[maxAction] = Pi[i][maxAction] + optimalActionIncrease
        # print(np.sum(Pi[i]), Pi[i], newPi)
        Pi[i] = newPi

        # Single sweep policy evaluation
        actionReward = 0
        # For max action sum expected reward of all possible (next state, reward) pairs
        for j in range(16):
            actionReward += Pr[i][maxAction][j] * (R[j] + gamma * V[j])
        Vnexts[i] = actionReward

        delta = max(delta, abs(Vnexts[i] - V[i]))

    V = np.copy(Vnexts)

    print("Pi", epoch)
    printPi(Pi)

    print("Policy evaluation", epoch, delta)
    printState(V)

    if(delta < theta):
        break