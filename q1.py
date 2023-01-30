import numpy as np
from numpy import random

results = np.zeros(shape=(100,50))
# Exploration constant
c = 0.01


def FindMaxAction(Q, t):
    maxValue = -float("inf")
    maxIndex = 0
    for i in range(10):
        Ucb = 0
        # Upper confidence bound
        if N[i] != 0:
            Ucb = Q[i] + c * np.sqrt(np.log(t) / N[i])
        else:
            # If Nt(a) = 0, then a is considered to be a maximizing action
            Ucb = float("inf")
        if(Ucb > maxValue):
            maxValue = Ucb
            maxIndex = i
    return maxIndex

for epoch in range(0,100):
    numOptimalActionChosen = 0
    totalReward = 0

    q = random.rand(10)
    trueOptimalAction = np.argmax(q)
            
    Q = np.zeros(10)
    N = np.zeros(10)
    
    # Interactions with the environment
    for i in range(1, 5001):
        # find action that maximizes UCB
        At = FindMaxAction(Q, i)
        
        # Try action, record reward from environment
        N[At] += 1
        if(At == trueOptimalAction):
            numOptimalActionChosen += 1
        v = random.rand()
        Rt = v < q[At]
        totalReward += Rt

        # Integrate reward into Q
        Q[At] = Q[At] + 1/N[At] * (Rt - Q[At])
        # print(At, Rt, Q)
        if(i % 100 == 0):
            print(epoch, i, "optimal action: ", numOptimalActionChosen, ", average reward: ", totalReward / i)
            results[epoch, i // 100 - 1] = totalReward / i

# print("true value", q, "\nagent estimate", Q)
# print(results[0],results[50],results[99])
# np.savetxt("results.csv", results, delimiter=",")
