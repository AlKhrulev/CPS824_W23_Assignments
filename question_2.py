from __future__ import annotations #used for type hints
import numpy as np


def run_automata(
    alpha: float, beta: float = 0, epoch_num: int = 5000
) -> tuple(float, list[float]):
    """
    A function that trains a single automata epoch_num many times,
    depending on values of alpha, beta.
    Every 100th epoch the proportion of cases when the optimal action was taken
    out of epoch_num cases will be printed, along with the overall average reward 
    (a sum of Q_t[a] for every action a)

    Args:
        alpha (float): a reward multiplier parameter
        beta (float, optional): a penalty multiplier parameter. Defaults to 0.
        epoch_num (int, optional): a number of epochs to train the automata for.
            Defaults to 5000.

    Returns:
        tuple(float, list[float]): a tuple containing the proportion for the last
            epoch and a numpy float array containing proportions for every 100th
            epoch(useful for training visualization)
    """
    Q_t = np.zeros(10, dtype=np.float64)  # accumulated rewards for each action
    N_t = np.zeros(
        10, dtype=np.uint16
    )  # number of times each action was taken(up to 65535 max)

    # an array of actual probabilities~continuous U(0,1)
    probabilities = np.random.rand(10)
    # the index=action that has the highest prob. of success
    optimal_action = np.argmax(probabilities)

    # initialize the array of probabilities to 0.1
    p_t = np.full((10), fill_value=0.1, dtype=np.float64)
    total_optimal_num = 0  # num. of times the optimal action was taken
    # create the mask for faster indexing
    mask = np.zeros_like(p_t, dtype=bool)

    # a helper variable that stores proportions for every epoch we print info for
    # can be useful to track the overall training progress
    proportions_per_selected_epoch = np.zeros(epoch_num // 100, dtype=np.float64)
    proportion_epoch_counter = 0  # a counter for numpy array

    print(f"{optimal_action=}")
    for epoch in range(epoch_num):

        # choose the action=array index from the list
        # of indices based on the probabilities p_t
        last_action = np.random.choice(np.arange(0, 10), p=p_t)
        # last_action=np.random.choice(np.arange(10))
        # print(f'last action is {last_action=}')
        # could have used rand()> probabilities[last_action] as well
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
            p_t[last_action] += alpha * (1 - p_t[last_action])
            p_t[~mask] *= 1 - alpha
        else:
            p_t[~mask] = beta / 9 + p_t[~mask] * (1 - beta)
            p_t[last_action] *= 1 - beta

        # just a rough check to make sure probabilities. sum to 1(should be redundant)
        np.testing.assert_almost_equal(p_t.sum(), 1.0)
        # reset the mask
        mask[:] = 0

        if epoch % 100 == 0:
            _proportion = total_optimal_num / epoch_num * 100
            print(f"FOR {epoch=}:\n-----------------")
            print(
                f"\tthe optimal action(={optimal_action}) was taken {total_optimal_num} times out of {epoch_num}={_proportion:.3f}%"
            )
            print(f"\tthe overall average reward is {Q_t.sum():.3f}\n\n")
            proportions_per_selected_epoch[proportion_epoch_counter] = _proportion
            proportion_epoch_counter += 1

    # print the result for the final epoch
    print(f"FOR the final epoch={epoch_num}:\n-----------------")
    print(
        f"\tthe optimal action(={optimal_action}) was taken {total_optimal_num} times out of {epoch_num}={total_optimal_num / epoch_num * 100:.3f}%"
    )
    print(f"\tthe overall average reward is {Q_t.sum():.3f}\n\n")

    # (used only for debugging)
    # print(p_t, N_t, Q_t, sep="\n")
    # print(f"initial values of {probabilities=}")

    return total_optimal_num / epoch_num * 100, proportions_per_selected_epoch


def _run_simulation(
    alpha_list: list[float],
    beta_list: list[float],
    epoch_num: int = 5000,
    repeats: int = 5,
):
    """
    A function that trains each automata for epoch_num iterations
    repeats many times. This is done to reduce the bias due to a randomly
    generated initial probabilities array. The alpha, beta parameters are
    taken from consecutive positions of their arrays. For ex., for
    alpha=[0,1], beta=[1,0] 2 automata will be trained, with
    alpha=0, beta=1, and with alpha=1, beta=0.
    The final output is written to the file "q2_alpha_beta_proportion.txt".

    Args:
        alpha_list (list[float]): a list of alphas to use
        beta_list (list[float]): a list of betas to use
        epoch_num (int, optional): the iteration number to
        train every automata. Defaults to 5000.
        repeats (int, optional): a number of times
        each automata must be trained for epoch_num iterations. Defaults to 5.
    """
    data = np.zeros((len(alpha_list) * repeats, 3), dtype=float)
    i = 0

    for alpha, beta in zip(alpha_list, beta_list):
        for j in range(repeats):
            # interested only in the final proportion, so discard the rest
            proportion, _ = run_automata(alpha, beta, epoch_num)
            data[i] = np.array([alpha, beta, proportion])
            i += 1

    np.savetxt("q2_alpha_beta_proportion.txt", data, fmt="%.3f", delimiter=",")


def _get_convergence_timings(
    alpha_list: list[float],
    beta_list: list[float],
    epoch_num: int = 5000,
    repeats: int = 1,
):
    """
    A function that trains each automata for epoch_num iterations
    repeats many times.  The alpha, beta parameters are
    taken from consecutive positions of their arrays. For ex., for
    alpha=[0,1], beta=[1,0] 2 automata will be trained, with
    alpha=0, beta=1, and with alpha=1, beta=0.
    The key thing is that it stores the proportions per every 100th epoch, making it
    possible to analyze the convergence rate of every model.
    The final output is written to the file "q2_alpha_beta_proportion_per_epoch.txt".

    Args:
        alpha_list (list[float]): a list of alphas to use
        beta_list (list[float]): a list of betas to use
        epoch_num (int, optional): the iteration number to
        train every automata. Defaults to 5000.
        repeats (int, optional): a number of times
        each automata must be trained for epoch_num iterations. Defaults to 5.
    """
    # alpha,beta,epoch_num_proportion = col number
    # for every alpha*repeats*epoch_num get a timing
    data = np.zeros((len(alpha_list) * repeats * (epoch_num // 100), 4), dtype=float)
    i = 0

    for alpha, beta in zip(alpha_list, beta_list):
        for j in range(repeats):
            # ignore the final proportion
            _, proportions = run_automata(alpha, beta, epoch_num)
            for epoch_index, proportion_per_epoch in enumerate(proportions):
                data[i] = np.array([alpha, beta, epoch_index, proportion_per_epoch])
                i += 1

    np.savetxt(
        "q2_alpha_beta_proportion_per_epoch.txt", data, fmt="%.3f", delimiter=","
    )


if __name__ == "__main__":
    # CONSTANTS
    EPOCH_NUM = 5000
    ALPHA = 0.015
    BETA = 0.01

    # an example of running a single training
    run_automata(ALPHA, BETA, EPOCH_NUM)

    # set parameters for training
    alpha_list=[0.01, 0.05, 0.1, 0.3, 0.1, 0.1, 0.1]
    beta_list = [0] * 4 + [0.01, 0.1, 0.2]
    # get final proportions and write to the disk
    _run_simulation(
        alpha_list=alpha_list, beta_list=beta_list, epoch_num=EPOCH_NUM
    )
    # # get proportions for each 100th epoch and write data to the disk
    _get_convergence_timings(
        alpha_list=alpha_list,
        beta_list=beta_list,
        repeats=1,
        epoch_num=EPOCH_NUM,
    )
