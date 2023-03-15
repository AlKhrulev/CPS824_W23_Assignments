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

# return if action at state is valid, i.e. will it collide with the walls
def canMove(state, action):
    nextState = (state[0] + actionDirection[action][0], state[1] + actionDirection[action][1])
    actionMagnitude = [-1,1,-1,1]
    barrierCheckPos = state
    if(actionMagnitude[action] > 0):
        barrierCheckPos = nextState
    
    valid = False
    if(action < 2):
        # vertical movement check horizontal
        valid = horizontalBarriers[barrierCheckPos[1]][barrierCheckPos[0]] == 0
    else:
        # horizontal movement check vertical
        valid = verticalBarriers[barrierCheckPos[1]][barrierCheckPos[0]] == 0

    # print(state, nextState)
    # print(barrierCheckPos, valid)
    return valid

print(canMove((5,4), 0))