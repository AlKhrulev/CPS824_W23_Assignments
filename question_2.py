import numpy as np


# CONSTANTS
EPOCH_NUM = 5000
ALPHA = 0.1
BETA = 0


# up to 65535 max
Q_t = np.zeros(10, dtype=np.uint16)  # accumulated rewards for each action
N_t = np.zeros(10, dtype=np.uint16)  # number of times each action was taken
# probabilities=np.random.dirichlet(np.ones(10),size=1).reshape(10,)

# an array of actual probabilities~continuous U(0,1)
probabilities = np.random.rand(10)
# cache response based on the probabilities(for fast selection)
response = (probabilities >= 0.5).astype(np.uint8)


# TODO del
# initialize the initial prob vector. Dirichlet ensures the prob. add up to 1
# p_t=np.random.dirichlet(np.ones(10),size=1).reshape(10,)

# initialize the array of probabilities to 0.1
p_t = np.full((10), fill_value=0.1, dtype=float)
total_optimal_num = 0  # num. of times the optimal action was taken
print(p_t, probabilities, response)

# create the mask for faster indexing
mask = np.zeros_like(p_t, dtype=bool)

for epoch in range(EPOCH_NUM):
    last_action = np.random.randint(0, 10, dtype=np.uint8)
    last_signal = response[last_action]
    optimal_action = np.argmax(Q_t)

    # modify total optimal number if the last action was indeed optimal
    total_optimal_num += (optimal_action == last_action)

    # update the action counter
    N_t[last_action] += 1
    # update the accumulated reward for that action
    # TODO check if this update is correct
    Q_t[last_action] += last_signal

    # set the last action to 1 for future selection
    mask[last_action] = 1

    if last_signal == 1:
        p_t[last_action] += ALPHA * (1 - p_t[last_action])
        p_t[~mask] *= 1 - ALPHA
    else:
        p_t[~mask] = BETA / 9 + p_t[~mask] * (1 - BETA)
        p_t[last_action] *= 1 - BETA

    # just a rough check to make sure probabilities. sum to 1(should be redundant)
    np.testing.assert_almost_equal(p_t.sum(), 1.0)
    # reset the mask
    mask[:] = 0

    if epoch % 100 == 0:
        print(f"FOR {epoch=}:\n-----------------")
        print(f"the optimal action was taken {total_optimal_num} times")
        print(f"the average reward is {Q_t}\n\n")

# TODO delete(used only for debugging)
print(p_t, N_t, Q_t, sep="\n")
