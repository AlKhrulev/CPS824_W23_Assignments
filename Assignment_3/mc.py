import numpy as np

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

p1 = 0.8
p2 = 0.1

# Actions A = [up, down, left, right]
actionDirection = [(0,-1), (0,1), (-1, 0), (1, 0)]
# Adjacent cells for each action
adjacentOffset = [[2, 3], [2, 3], [0, 1], [0, 1]]

def addTuples(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])

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
    print("Next state probabilities", Pr)

    # Pick a next state from Pr
    keys = list(Pr.keys())
    choice = np.random.choice(len(keys), 1, p=list(Pr.values()))[0]
    nextState = keys[choice]
    # reward = nextState
    return nextState

print(interactEnvironment((5,4), 0))
print(interactEnvironment((5,4), 1))
print(interactEnvironment((5,4), 2))
print(interactEnvironment((5,4), 3))
print(interactEnvironment((2,2), 0))
print(interactEnvironment((2,2), 1))
print(interactEnvironment((2,2), 2))
print(interactEnvironment((2,2), 3))