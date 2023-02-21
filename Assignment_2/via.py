# Alexander Khrulev 500882732
# Mahan Pandey 500881861

from q3 import *
import time

# Value iteration
# Instead of waiting for policy evaluation to converge 
# Find action that maximizes reward in next step
# Update policy evaluation based on best max action
# Init V(s) with 0
V = np.zeros((16))
Vnexts = np.zeros(16)
epoch = 0
totalTimeStart = time.time()
while(True):
    startTime = time.time()
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
        # print(np.sum(Pi[i]), Pi[i], newPi)
        Pi[i] = newPi

        # Single sweep policy evaluation = action reward of max action
        Vnexts[i] = maxReward

        delta = max(delta, abs(Vnexts[i] - V[i]))

    V = np.copy(Vnexts)

    # print("Pi", epoch)
    # printPi(Pi)

    # print("Policy evaluation", epoch, delta)
    # printState(V)

    print(f"Iteration {epoch} took {(time.time() - startTime)*1000}ms")

    if(delta < theta):
        break


print(f"Value Iteration Pi from {epoch} iterations in {(time.time() - totalTimeStart)*1000}ms")
printPi(Pi)