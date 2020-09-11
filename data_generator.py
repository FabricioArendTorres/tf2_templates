import numpy as np
from typing import Tuple


def generate_sinus_data(N, seed=123):
    r"""

    :param N:
    :param seed:
    :return: X, Y
    """
    rng = np.random.RandomState(seed)

    def func(x):
        return 3 * x + np.sin(x * 3 * 3.14)  # + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

    X = rng.rand(N, 1) * 4 - 2  # X values
    Y = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values

    # Xt = np.linspace(-2.1, 2.1, 1000)[:, None]
    # Yt = func(Xt)
    # _ = plt.plot(Xt, Yt, c="k")
    # plt.show()
    return X, Y


def generate_banana_data(n_sqrt_train: int, n_sqrt_test: int, seed: int = None, type='float32') -> Tuple[
    Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    """

    Args:
        n_sqrt_train: create n training points, on a sqrt(n)*sqrt(n) randomly spaced grid
        n_sqrt_test:  create n test points, on a sqrt(n)*sqrt(n) equidistant grid
        seed: seeding
        type: data type

    Returns: (X,Y), (X_test, Y_test)

    """

    def generate_data(N, a=1, b=100,
                             x_min=-2, x_max=2, y_min=-2, y_max=4,
                             linspace=False):
        rng = np.random.RandomState(seed)

        fun = lambda x, y: np.log((a - x) ** 2 + b * (y - x ** 2) ** 2 + 1)

        if not linspace:
            xdim_seq = np.sort(rng.uniform(x_min, x_max, N))
            ydim_seq = np.sort(rng.uniform(y_min, y_max, N))
        else:
            xdim_seq = np.linspace(x_min, x_max, N)
            ydim_seq = np.linspace(y_min, y_max, N)

        xx, yy = np.meshgrid(xdim_seq, ydim_seq)
        zz = fun(xx, yy)
        return xx.astype(type), yy.astype(type), zz.astype(type)

    xx, yy, zz = generate_data(n_sqrt_train, linspace=False)
    xx_t, yy_t, zz_t = generate_data(n_sqrt_test, linspace=True)

    X = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    Y = zz.reshape(-1, 1)

    X_test = np.concatenate([xx_t.reshape(-1, 1), yy_t.reshape(-1, 1)], axis=1)
    Y_test = zz_t.reshape(-1, 1)
    return (X, Y), (X_test, Y_test)


def generate_hmm_data(data_count, time_count):
    """Signal-and-Noise HMM dataset
    Obtained from https://github.com/dtak/tree-regularization-public
    The generative process comes from two separate HMM processes. First,
    a "signal" HMM generates the first 7 data dimensions from 5 well-separated states.
    Second, an independent "noise" HMM generates the remaining 7 data dimensions
    from a different set of 5 states. Each timestep's output label is produced by a
    rule involving both the signal data and the signal hidden state.
    @param data_count: number of sequences in dataset
    @param time_count: number of timesteps in a sequence
    @return obs_set: Torch Tensor data_count x time_count x 14
    @return out_set: Torch Tensor data_count x time_count x 1
    """

    bias_mat = np.array([15])
    # 5 states + 7 observations
    weight_mat = np.array([[10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0]])

    state_count = 5
    dim_count = 7
    out_count = 1

    # signal HMM process
    pi_mat_signal = np.array([.5, .5, 0, 0, 0])
    trans_mat_signal = np.array(([.7, .3, 0, 0, 0],
                                 [.5, .25, .25, 0, 0],
                                 [0, .25, .5, .25, 0],
                                 [0, 0, .25, .25, .5],
                                 [0, 0, 0, .5, .5]))
    obs_mat_signal = np.array(([.5, .5, .5, .5, 0, 0, 0],
                               [.5, .5, .5, .5, .5, 0, 0],
                               [.5, .5, .5, 0, .5, 0, 0],
                               [.5, .5, .5, 0, 0, .5, 0],
                               [.5, .5, .5, 0, 0, 0, .5]))

    # noise HMM process
    pi_mat_noise = np.array([.2, .2, .2, .2, .2])
    trans_mat_noise = np.array(([.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2]))
    obs_mat_noise = np.array(([.5, .5, .5, 0, 0, 0, 0],
                              [0, .5, .5, .5, 0, 0, 0],
                              [0, 0, .5, .5, .5, 0, 0],
                              [0, 0, 0, .5, .5, .5, 0],
                              [0, 0, 0, 0, .5, .5, .5]))

    # create the sequences
    obs_set = np.zeros((dim_count * 2, time_count, data_count))
    out_set = np.zeros((out_count, time_count, data_count))

    state_set_signal = np.zeros((state_count, time_count, data_count))
    state_set_noise = np.zeros((state_count, time_count, data_count))

    # loop through to sample HMM states
    for data_ix in range(data_count):
        for time_ix in range(time_count):
            if time_ix == 0:
                state_signal = np.random.multinomial(1, pi_mat_signal)
                state_noise = np.random.multinomial(1, pi_mat_noise)
                state_set_signal[:, 0, data_ix] = state_signal
                state_set_noise[:, 0, data_ix] = state_noise
            else:
                tvec_signal = np.dot(state_set_signal[:, time_ix - 1, data_ix], trans_mat_signal)
                tvec_noise = np.dot(state_set_noise[:, time_ix - 1, data_ix], trans_mat_noise)
                state_signal = np.random.multinomial(1, tvec_signal)
                state_noise = np.random.multinomial(1, tvec_noise)
                state_set_signal[:, time_ix, data_ix] = state_signal
                state_set_noise[:, time_ix, data_ix] = state_noise

    # loop through to generate observations and outputs
    for data_ix in range(data_count):
        for time_ix in range(time_count):
            obs_vec_signal = np.dot(state_set_signal[:, time_ix, data_ix], obs_mat_signal)
            obs_vec_noise = np.dot(state_set_noise[:, time_ix, data_ix], obs_mat_noise)
            obs_signal = np.random.binomial(1, obs_vec_signal)
            obs_noise = np.random.binomial(1, obs_vec_noise)
            obs = np.hstack((obs_signal, obs_noise))  # concat together
            obs_set[:, time_ix, data_ix] = obs

            # input is state concatenated with observation
            in_vec = np.hstack((state_set_signal[:, time_ix, data_ix],
                                obs_set[:dim_count, time_ix, data_ix]))

            # output is a logistic regression on W \dot input
            out_vec = 1 / (1 + np.exp(-1 * (np.dot(weight_mat, in_vec) - bias_mat)))

            out = np.random.binomial(1, out_vec)
            out_set[:, time_ix, data_ix] = out

    return obs_set.T, out_set.T
