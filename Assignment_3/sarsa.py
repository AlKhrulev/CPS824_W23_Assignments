from __future__ import annotations
import numpy as np
from typing import final, Literal

# 10x10 grid world

# store barriers in separate arrays for each axis
# horizontalBarriers 11x10
horizontalBarriers = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# 10x11
verticalBarriers = [
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
]

p1 = 0.8
p2 = 0.1


# Actions A = [up, down, left, right]
actionDirection = [(0, -1), (0, 1), (-1, 0), (1, 0)]
# Adjacent cells for each action
adjacentOffset = [[2, 3], [2, 3], [0, 1], [0, 1]]


def addTuples(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])


# return if action at state is valid, i.e. will it collide with the walls
def canMove(state, actionDirection):
    nextState = addTuples(state, actionDirection)
    actionMagnitude = actionDirection[0] + actionDirection[1]
    barrierCheckPos = state
    if actionMagnitude > 0:
        barrierCheckPos = nextState

    valid = False
    if actionDirection[0] == 0:
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

    if canMove(s, actionDirection[a]):
        # IntendedNextState valid
        intendedNextState = addTuples(s, actionDirection[a])
        Pr[intendedNextState] = p1
        Pr[s] = p2

        if canMove(intendedNextState, adDir1):
            ad1 = addTuples(intendedNextState, adDir1)
            Pr[ad1] = adjacentPr
        else:
            # if adjacent not within bounds, "probability of moving in the desired direction is a bit higher" (Q10)
            Pr[intendedNextState] += adjacentPr

        if canMove(intendedNextState, adDir2):
            ad2 = addTuples(intendedNextState, adDir2)
            Pr[ad2] = adjacentPr
        else:
            # if adjacent not within bounds, "probability of moving in the desired direction is a bit higher" (Q10)
            Pr[intendedNextState] += adjacentPr
    else:
        # intendedNextState not valid
        Pr[s] = p1 + p2

        if canMove(s, adDir1):
            ad1 = addTuples(s, adDir1)
            Pr[ad1] = adjacentPr
        else:
            # if adjacent not within bounds, probability of staying at s higher
            Pr[s] += adjacentPr

        if canMove(s, adDir2):
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
    reward = 100 if (nextState[0] == 9 and nextState[1] == 0) else -1
    return (nextState, reward)


print(interactEnvironment((5, 4), 0))


def select_action(current_state_Q, epsilon: float) -> Literal[0, 1, 2, 3]:
    # Actions A = [up, down, left, right]
    # actionDirection = [(0,-1), (0,1), (-1, 0), (1, 0)]

    # if randomly generated number is less than epsilon,
    # do a random action
    print(f"{current_state_Q=}")
    if np.random.rand() < epsilon:
        print("in epsilon")
        return np.random.randint(0, 5)

    # otherwise, return the action that maximizes Q at the current state
    print("in argmax")
    return np.argmax(current_state_Q)


if __name__ == "__main__":
    # the terminal state
    TERMINAL_STATE: tuple[int, int] = (0, 9)
    # the exploration constant
    EPSILON: final[float] = 0.1
    # the learning rate
    ALPHA: final[float] = 0.3
    # the discount constant
    GAMMA: final[float] = 0.3
    # the number of episodes to run
    TOTAL_EPISODE_NUMBER: final[int] = 1
    Q = np.zeros((10, 10, 4), dtype=np.float64)

    for episode_number in range(TOTAL_EPISODE_NUMBER):
        # generate a random initial state as a tuple of r.v. from
        # 0 to 9
        current_state = tuple(np.random.randint(low=0, high=10, size=2))
        print(current_state, type(current_state))
        current_action: Literal[0, 1, 2, 3] = select_action(Q[current_state], EPSILON)
        print(f"{current_action=}")

        while current_state != TERMINAL_STATE:
            print("debug")
            next_state: tuple[int, int]
            immediate_reward: Literal[-1, 100]  # the immed. reward is either -1 or 100
            # find the next state
            next_state, immediate_reward = interactEnvironment(
                current_state, current_action
            )

            # choose A' from S' using policy derrived from Q
            next_action: Literal[0, 1, 2, 3] = select_action(Q[next_state[0]], EPSILON)
            print(f"{next_action=},{next_state=}")
            # update Q
            Q[current_state][current_action] = Q[current_state][
                current_action
            ] + ALPHA * (
                immediate_reward
                + GAMMA * Q[next_state][next_action]
                - Q[current_state][current_action]
            )
            # update actions and a state
            current_action, current_state = next_action, next_state
