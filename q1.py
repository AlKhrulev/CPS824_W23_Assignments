import numpy
from numpy import random


numOptimalActionChosen = 0
totalReward = 0

q = random.rand(10)
trueOptimalAction = 0
maxValue = -float("inf")
for i in range(10):
    if(q[i] > maxValue):
        maxValue = q[i]
        trueOptimalAction = i
        
Q = [0] * 10
N = [0] * 10
# Exploration constant
c = 10

def TryAction(action):
    global numOptimalActionChosen, totalReward
    N[action] += 1
    if(action == trueOptimalAction):
        numOptimalActionChosen += 1
    v = random.rand()
    # print("TryAction", v, q[action])
    Rt = v < q[action]
    totalReward += Rt
    return Rt

def FindMaxAction(t):
    maxValue = -float("inf")
    maxIndex = 0
    for i in range(10):
        Ucb = 0
        # Upper confidence bound
        if N[i] != 0:
            Ucb = Q[i] + c * numpy.sqrt(numpy.log(t) / N[i])
        else:
            # If Nt(a) = 0, then a is considered to be a maximizing action
            Ucb = float("inf")
        if(Ucb > maxValue):
            maxValue = Ucb
            maxIndex = i
    return maxIndex

# Interactions with the environment
for i in range(0, 5000):
    # find action that maximizes UCB
    At = FindMaxAction(i)
    # Try action, record reward from environment
    Rt = TryAction(At)
    Q[At] = Q[At] + 1/N[At] * (Rt - Q[At])
    # print(At, Rt, Q)
    if(i % 100 == 0):
        print(i, "optimal action chosen: ", numOptimalActionChosen, ", average reward over time: ", totalReward / i)

# print("true value", q, "\nagent estimate", Q)
# print(numOptimalActionChosen)