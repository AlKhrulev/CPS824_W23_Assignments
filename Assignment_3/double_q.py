# Alexander Khrulev 500882732
# Mahan Pandey 500881861

import numpy as np
from numpy import random
from time import perf_counter

p1 = 1
p2 = 0

gamma = 0.9
epsilon = 0.1
alpha = 0.1

# 10x10 grid world
# store barriers in separate arrays for each axis
# horizontalBarriers 11x10
horizontalBarriers =   [[1,1,1,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [1,1,0,1,1,1,1,0,1,1],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [1,1,1,1,1,1,1,1,1,1]]

# 10x11
verticalBarriers =     [[1,0,0,0,0,1,0,0,0,0,1],
                        [1,0,0,0,0,1,0,0,0,0,1],
                        [1,0,0,0,0,0,0,0,0,0,1],
                        [1,0,0,0,0,1,0,0,0,0,1],
                        [1,0,0,0,0,1,0,0,0,0,1],
                        [1,0,0,0,0,1,0,0,0,0,1],
                        [1,0,0,0,0,1,0,0,0,0,1],
                        [1,0,0,0,0,0,0,0,0,0,1],
                        [1,0,0,0,0,1,0,0,0,0,1],
                        [1,0,0,0,0,1,0,0,0,0,1]]

# Actions A = [up, down, left, right]
actionDirection = [(0,-1), (0,1), (-1, 0), (1, 0)]
# Adjacent cells for each action
adjacentOffset = [[2, 3], [2, 3], [0, 1], [0, 1]]

def addTuples(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])

def printActionValue(Q):
    for a in range(4):
        print(f"Action: {a}")
        for y in range(10):
            for x in range(10):
                # print("(%2d" % (x + y*4) + ", %.2f" % s[x + y * 4] + ")", end='')
                print("| % 3.2f" % Q[x][y][a], end='')
            print()

def printC(C):
    for a in range(4):
        print(f"Action: {a}")
        for y in range(10):
            for x in range(10):
                # print("(%2d" % (x + y*4) + ", %.2f" % s[x + y * 4] + ")", end='')
                print("| % 3.0f" % C[x][y][a], end='')
            print()

def printPi(Q):
    arctionArrow = [" ↑ ", " ↓ ", " ← ", " → "]
    for y in range(10):
        for x in range(10):
            optimalAction = maxAction(Q, (x,y))
            if(x == 9 and y == 0):
                print(" x |", end='')
            else:
                # print(str(actionDirection[optimalAction]) + "|", end='')
                print(arctionArrow[optimalAction] + "|", end='')
                # print("%.2f" % Pi[x][y][optimalAction] + "|", end='')
                # print("%.2f" % Pi[x][y][0] + " %.2f" % Pi[x][y][1] + " %.2f" % Pi[x][y][2] + " %.2f" % Pi[x][y][3]+ "|", end='')

        print()

# return if action at state is valid, i.e. will it collide with the walls
def canMove(state, actionDirection):
    nextState = addTuples(state, actionDirection)
    actionMagnitude = actionDirection[0] + actionDirection[1]
    barrierCheckPos = state
    if(actionMagnitude > 0):
        barrierCheckPos = nextState
    
    valid = False
    if(actionDirection[0] == 0):
        # vertical movement check horizontal
        valid = horizontalBarriers[barrierCheckPos[1]][barrierCheckPos[0]] == 0
    else:
        # horizontal movement check vertical
        valid = verticalBarriers[barrierCheckPos[1]][barrierCheckPos[0]] == 0

    # print(state, nextState)
    # print(barrierCheckPos, valid)
    return valid

# state, action -> next state, reward
def interactEnvironment(s, a):
    # Adjacent cells for action
    adjacentPr = (1 - p1 - p2) / 2
    adDir1 = actionDirection[adjacentOffset[a][0]]
    adDir2 = actionDirection[adjacentOffset[a][1]]

    # Next state probabilities
    Pr = {}

    if (canMove(s, actionDirection[a])):
        # IntendedNextState valid
        intendedNextState = addTuples(s, actionDirection[a])
        Pr[intendedNextState] = p1
        Pr[s] = p2

        if(canMove(intendedNextState, adDir1)):
            ad1 = addTuples(intendedNextState, adDir1)
            Pr[ad1] = adjacentPr
        else:
            # if adjacent not within bounds, "probability of moving in the desired direction is a bit higher" (Q10)
            Pr[intendedNextState] += adjacentPr

        if(canMove(intendedNextState, adDir2)):
            ad2 = addTuples(intendedNextState, adDir2)
            Pr[ad2] = adjacentPr
        else:
            # if adjacent not within bounds, "probability of moving in the desired direction is a bit higher" (Q10)
            Pr[intendedNextState] += adjacentPr
    else:
         # intendedNextState not valid
        Pr[s] = p1 + p2

        if(canMove(s, adDir1)):
            ad1 = addTuples(s, adDir1)
            Pr[ad1] = adjacentPr
        else:
            # if adjacent not within bounds, probability of staying at s higher
            Pr[s] += adjacentPr

        if(canMove(s, adDir2)):
            ad2 = addTuples(s, adDir2)
            Pr[ad2] = adjacentPr
        else:
            # if adjacent not within bounds, probability of staying at s higher
            Pr[s] += adjacentPr
    # print("Next state probabilities", Pr)

    # Pick a next state from Pr
    keys = list(Pr.keys())
    choice = random.choice(len(keys), 1, p=list(Pr.values()))[0]
    nextState = keys[choice]
    reward = 100 if (nextState[0] == 9 and nextState[1] == 0) else -1
    return (nextState, reward)

def maxAction(Q, s):
    sx = s[0]
    sy = s[1]
    actionValues = [Q[sx][sy][0], Q[sx][sy][1], Q[sx][sy][2], Q[sx][sy][3]]
    return np.argmax(actionValues) 

# epsilon-soft
def chooseEGreedyAction(Q1, Q2, s):
    if random.rand() < epsilon:
        return random.randint(4)
    else:
        sx = s[0]
        sy = s[1]
        actionValues = [Q1[sx][sy][0] + Q2[sx][sy][0], Q1[sx][sy][1] + Q2[sx][sy][1], Q1[sx][sy][2] + Q2[sx][sy][2], Q1[sx][sy][3] + Q2[sx][sy][3]]
        return np.argmax(actionValues) 

# average first time visit return for action
Q1 = np.zeros((10,10,4))
Q2 = np.zeros((10,10,4))

# C count of fist time visits to state
C = np.zeros((10,10,4))
episodes = 30000
totalIterations = 0
start_time = perf_counter()

for i in range(episodes):
    states = []
    actions = []
    rewards = []

    # Generate episode
    randPos = random.randint(0, 10, 2)
    s = (randPos[0], randPos[1])
    while True:
        totalIterations += 1
        # Choose e-greedy action according to Q1 + Q2
        a = chooseEGreedyAction(Q1, Q2, s)
        env = interactEnvironment(s, a)
        sx = s[0]
        sy = s[1]
        reward = env[1]
        nextS = env[0]

        if random.rand() < 0.5:
            maxQ2 = Q2[nextS[0]][nextS[1]][maxAction(Q1, nextS)]
            Q1[sx][sy][a] = Q1[sx][sy][a] + alpha * (reward + gamma * maxQ2 - Q1[sx][sy][a])
        else:
            maxQ1 = Q1[nextS[0]][nextS[1]][maxAction(Q2, nextS)]
            Q2[sx][sy][a] = Q2[sx][sy][a] + alpha * (reward + gamma * maxQ1 - Q2[sx][sy][a])
        
        s = nextS
        # terminate on terminal state
        if reward == 100:
            break

print(f"Double Q-learning finished in {perf_counter()-start_time:.3f}s with {episodes} episodes, {totalIterations} total iterations")
printPi(Q1)
# printActionValue(Q)
# printC(C)