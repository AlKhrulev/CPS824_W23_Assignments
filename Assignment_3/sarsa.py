from __future__ import annotations
import numpy as np
from typing import final, Literal
from time import perf_counter

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
def interactEnvironment(s, a, p1, p2):
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
    # print("Next state probabilities", Pr)

    # Pick a next state from Pr
    keys = list(Pr.keys())
    choice = np.random.choice(len(keys), 1, p=list(Pr.values()))[0]
    nextState = keys[choice]
    reward = 100 if (nextState[0] == 9 and nextState[1] == 0) else -1
    return (nextState, reward)


# print(interactEnvironment((5, 4), 0))


def select_epsilon_greedy_action(
    current_state_Q, epsilon: float
) -> Literal[0, 1, 2, 3]:
    """
    Select a random action based on the epsilon-greedy policy.
    With the prob. of epsion will return a random action;
    Otherwise, return the action that maximizes the state-value estimate for
    the current state.
    The set of action is A = [up, down, left, right] and their directions are
    actionDirection = [(0,-1), (0,1), (-1, 0), (1, 0)].
    The function will return a value from 0 to 3 corresponding to the index
    of action from A.
    """
    # if randomly generated number is less than epsilon,
    # do a random action
    if np.random.rand() < epsilon:
        return np.random.randint(0, 4)

    # otherwise, return the action that maximizes Q at the current state
    return np.argmax(current_state_Q)


def _visualize_Q(Q, terminal_state):
    action_icons = ("↑", "↓", "←", "→")
    # array containing maximizing actions for each cell
    maximizing_actions = np.argmax(Q, axis=2)
    pretty_print = np.zeros_like(maximizing_actions, dtype=np.object_)
    # represent the maximizing actions as arrows
    pretty_print[maximizing_actions == 0] = action_icons[0]
    pretty_print[maximizing_actions == 1] = action_icons[1]
    pretty_print[maximizing_actions == 2] = action_icons[2]
    pretty_print[maximizing_actions == 3] = action_icons[3]
    # set the terminal state mark to X
    pretty_print[terminal_state] = "X"

    # Need to transpose both arrays to have the
    # terminal state in the right most corner.
    # print("The maximum state-action values for each state:")
    # the maximum state-action values for each state
    # print(np.max(Q, axis=2).T)
    print("The optimal policy is")
    # the actions that maximize state-action values
    print(pretty_print.T)


if __name__ == "__main__":
    """
    IMPORTANT: with this implementation, the terminal state ends up being (9,0)
    while numpy's Q is oriented the normal way(row then column-wise).
    As a result, Q is actually a transpose of a picture
    given in the class. That's why _visualize_Q() uses transpose to restore the
    orientation. This affects only the visual part, as the picture is fully
    symmetric.
    """
    # input p1, p2
    p1 = float(input("Please enter p1:\n"))
    p2 = float(input("Please enter p2:\n"))

    # the terminal state
    TERMINAL_STATE: tuple[int, int] = (9, 0)
    # the exploration constant
    EPSILON: final[float] = 0.1
    # the learning rate
    ALPHA: final[float] = 0.1
    # the discount constant
    GAMMA: final[float] = 0.9
    # the number of episodes to run
    TOTAL_EPISODE_NUMBER: final[int] = 1000
    # the state-value function estimate array
    Q = np.zeros((10, 10, 4), dtype=np.float64)
    start_time = perf_counter()
    loop_num = 0

    for episode_number in range(TOTAL_EPISODE_NUMBER):
        # generate a random initial state as a tuple of r.v. from
        # 0 to 9
        current_state = tuple(np.random.randint(low=0, high=10, size=2))
        print(f"the initial state is {current_state}")
        current_action: Literal[0, 1, 2, 3] = select_epsilon_greedy_action(
            Q[current_state], EPSILON
        )
        print(f"the first action is {current_action}")

        while current_state != TERMINAL_STATE:
            # add a state to the set of explored states
            # print(f"debug for {loop_num=}")
            next_state: tuple[int, int]
            immediate_reward: Literal[-1, 100]  # the immed. reward is either -1 or 100
            # find the next state
            next_state, immediate_reward = interactEnvironment(
                current_state, current_action, p1, p2
            )

            # choose A' from S' using policy derrived from Q
            next_action: Literal[0, 1, 2, 3] = select_epsilon_greedy_action(
                Q[next_state], EPSILON
            )
            # print(f"{next_action=},{next_state=}")
            # update Q
            Q[current_state][current_action] = Q[current_state][
                current_action
            ] + ALPHA * (
                immediate_reward
                + GAMMA * Q[next_state][next_action]
                - Q[current_state][current_action]
            )
            # update current action and a state
            current_action, current_state = next_action, next_state
            loop_num += 1

        print(f"finished running {episode_number=}")

    print(
        f"""it took {perf_counter()-start_time:.3f} s to run {TOTAL_EPISODE_NUMBER=} episodes with a total of
        {loop_num} iterations"""
    )
    _visualize_Q(Q, TERMINAL_STATE)

    # Consider manually transposing Q here to aligh it with the image in the assigment for good
    # kind of like Q=Q.T but more complicated as it is a 3d array.
    # _visualize_Q takes care of that
