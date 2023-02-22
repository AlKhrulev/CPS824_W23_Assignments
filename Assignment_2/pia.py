# Alexander Khrulev 500882732
# Mahan Pandey 500881861

from q3 import *
import time

V = np.zeros((16))
Vnexts = np.zeros(16)
epoch = 0

totalTimeStart = time.time()
# Policy iteration
while(True):
    startTime = time.time()
    epoch += 1
    # Policy Evaluation
    while(True):
        delta = 0
        for i in range(16):
            lastV = V[i]
            # For each state sum expected reward of all possible actions
            nextV = 0
            for a in range(4):
                actionReward = 0
                # For each action sum expected reward of all possible (next state, reward) pairs
                for j in range(16):
                    actionReward += Pr[i][a][j] * (getReward(j, a) + gamma * V[j])
                nextV += Pi[i][a] * actionReward
            Vnexts[i] = nextV
            delta = max(delta, abs(nextV - lastV))
        
        V = np.copy(Vnexts)
        if(delta < theta):
            break

    # Policy Improvement
    policyStable = True
    for i in range(16):
        oldMaxAction = np.argmax(Pi[i])
        maxAction = 0
        maxReward = float('-inf')
        # For each state find action that maximizes expected reward
        for a in range(4):
            actionReward = 0
            # For each action sum expected reward of all possible (next state, reward) pairs
            for j in range(16):
                actionReward += Pr[i][a][j] * (getReward(j, a) + gamma * V[j])
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
        Pi[i] = newPi

    print(f"Iteration {epoch} took {(time.time() - startTime)*1000}ms")

    if policyStable:
        break

print(f"Policy Iteration Pi from {epoch} iterations in {(time.time() - totalTimeStart)*1000}ms")
printPi(Pi)