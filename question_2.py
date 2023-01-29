import numpy as np
import logging


# CONSTANTS
EPOCH_NUM = 10000
ALPHA = 0.015
BETA = 0.01


Q_t = np.zeros(10, dtype=np.float64)  # accumulated rewards for each action
N_t = np.zeros(
    10, dtype=np.uint16
)  # number of times each action was taken(up to 65535 max)
# probabilities=np.random.dirichlet(np.ones(10),size=1).reshape(10,)

# an array of actual probabilities~continuous U(0,1)
probabilities = np.random.rand(10)
# the index of action that has the highest prob. of success
optimal_action = np.argmax(probabilities)


# TODO del
# initialize the initial prob vector. Dirichlet ensures the prob. add up to 1
# p_t=np.random.dirichlet(np.ones(10),size=1).reshape(10,)

# initialize the array of probabilities to 0.1
p_t = np.full((10), fill_value=0.1, dtype=np.float64)
total_optimal_num = 0  # num. of times the optimal action was taken
print(p_t, probabilities)
# a helper variable to cache the indexing for future use
_indeces = np.arange(0, 10, dtype=np.uint8)
print(f"{_indeces=}")
# create the mask for faster indexing
mask = np.zeros_like(p_t, dtype=bool)

print(f"{optimal_action=}")
for epoch in range(EPOCH_NUM):

    # choose the action=array index from the list
    # of indeces based on the probabilties p_t
    last_action = np.random.choice(np.arange(0, 10), p=p_t)
    # last_action=np.random.choice(np.arange(10))
    # print(f'last action is {last_action=}')
    # generate a signal based on the q_t
    last_signal = np.random.choice(
        [1, 0], p=[probabilities[last_action], 1 - probabilities[last_action]]
    )

    # modify total optimal number if the last action was indeed optimal
    total_optimal_num += optimal_action == last_action

    # update the action counter
    N_t[last_action] += 1
    # update the accumulated reward for that action
    Q_t[last_action] += (1 / N_t[last_action]) * (last_signal - Q_t[last_action])

    # set the last action to 1 for future selection
    mask[last_action] = True

    if last_signal == 1:
        p_t[last_action] += ALPHA * (1 - p_t[last_action])
        p_t[~mask] *= 1 - ALPHA
    else:
        logging.debug(f"{p_t=} before change")
        logging.debug(f"{mask=} after set up")
        logging.debug(f"{p_t[~mask]=}")
        p_t[~mask] = BETA / 9 + p_t[~mask] * (1 - BETA)
        logging.debug(f"{p_t=} after ~mask")
        p_t[last_action] *= 1 - BETA
        logging.debug(f"{p_t=} after last action")
    # just a rough check to make sure probabilities. sum to 1(should be redundant)
    np.testing.assert_almost_equal(p_t.sum(), 1.0)
    # reset the mask
    mask[:] = 0

    if epoch % 100 == 0:
        print(f"FOR {epoch=}:\n-----------------")
        print(
            f"the optimal action was taken {total_optimal_num} times={total_optimal_num/EPOCH_NUM*100}%"
        )
        print(f"the average reward is {Q_t}\n\n")

# TODO delete(used only for debugging)
print(p_t, N_t, Q_t, sep="\n")
print(f"initial values of {probabilities=}")
