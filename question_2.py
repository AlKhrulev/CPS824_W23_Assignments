import numpy as np


def find_position(p_t, random_value=None):
    # TODO add more documentation
    # we generate/use a random number in the range from 0 to 1
    # sort the array
    # find the index where the random number would need to be inserted
    # to maintain the order. The prob. at that index is the one closest to ours.
    # Now, find the index of that closest probability in the original array.
    # That number will serve as a random action we want to take
    """
    Sample Usage:
    random_value=0.6341171045877046
p_t=array([0.19399856, 0.53831702, 0.00435142, 0.71462101, 0.57039106,
       0.79182353, 0.98224042, 0.78473985, 0.89165213, 0.69857165])
p_t_sorted=array([0.00435142, 0.19399856, 0.53831702, 0.57039106, 0.69857165,
       0.71462101, 0.78473985, 0.79182353, 0.89165213, 0.98224042])
position_in_sorted=4
closest_probability=0.6985716477516771
true_position=array([9], dtype=int64)
random_value=0.7159561528061424
p_t=array([0.19399856, 0.53831702, 0.00435142, 0.71462101, 0.57039106,
       0.79182353, 0.98224042, 0.78473985, 0.89165213, 0.69857165])
p_t_sorted=array([0.00435142, 0.19399856, 0.53831702, 0.57039106, 0.69857165,
       0.71462101, 0.78473985, 0.79182353, 0.89165213, 0.98224042])
position_in_sorted=6
closest_probability=0.7847398506606472
true_position=array([7], dtype=int64)
random_value=0.12892700807072133
p_t=array([0.19399856, 0.53831702, 0.00435142, 0.71462101, 0.57039106,
       0.79182353, 0.98224042, 0.78473985, 0.89165213, 0.69857165])
p_t_sorted=array([0.00435142, 0.19399856, 0.53831702, 0.57039106, 0.69857165,
       0.71462101, 0.78473985, 0.79182353, 0.89165213, 0.98224042])
position_in_sorted=1
closest_probability=0.1939985555898357
true_position=array([0], dtype=int64)
    """
    if not random_value:
        random_value = np.random.rand()
    p_t_sorted = np.sort(p_t)
    position_in_sorted = np.searchsorted(p_t_sorted, random_value)
    print(f'{position_in_sorted=}')
    closest_probability = p_t_sorted[position_in_sorted]
    true_position = np.where(np.isclose(p_t, closest_probability))[0]
    print(
        f"{random_value=}\n{p_t=}\n{p_t_sorted=}\n{position_in_sorted=}\n{closest_probability=}\n{true_position=}"
    )
    return true_position

# CONSTANTS
EPOCH_NUM = 30000
ALPHA = 0.1
BETA = 0


Q_t = np.zeros(10, dtype=np.float64)  # accumulated rewards for each action
N_t = np.zeros(
    10, dtype=np.uint16
)  # number of times each action was taken(up to 65535 max)
# probabilities=np.random.dirichlet(np.ones(10),size=1).reshape(10,)

# an array of actual probabilities~continuous U(0,1)
probabilities = np.random.rand(10)
# cache response based on the probabilities(for fast selection)
response = (probabilities >= 0.5).astype(np.uint8)
# the index of action that has the highest prob. of success
optimal_action = np.argmax(probabilities)


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
    
    # last_action = np.random.randint(0, 10, dtype=np.uint8)
    last_action= find_position(probabilities,None)
    last_signal = response[last_action]

    # modify total optimal number if the last action was indeed optimal
    total_optimal_num += optimal_action == last_action

    # update the action counter
    N_t[last_action] += 1
    # update the accumulated reward for that action
    # TODO check if this update is correct
    Q_t[last_action] += (1 / (epoch + 1)) * (last_action - Q_t[last_action])

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

    # if epoch % 100 == 0:
    if epoch == (EPOCH_NUM - 1):
        print(f"FOR {epoch=}:\n-----------------")
        print(f"the optimal action was taken {total_optimal_num} times")
        print(f"the average reward is {Q_t}\n\n")

# TODO delete(used only for debugging)
print(p_t, N_t, Q_t, sep="\n")



# for i in range(3):
#     find_position(probabilities,None)
