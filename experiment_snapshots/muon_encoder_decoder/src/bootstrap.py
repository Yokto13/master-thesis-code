import numpy as np


def bootstrapped_CI(groups: np.ndarray, n, ci) -> tuple[float, float]:
    """Compute bootstrapped confidence interval for the mean of data.

    How this works?
    Let X_1, ..., X_G by random variables representing score for G different games.
    For each game we have S realizations that are i.i.d, we thus have G*S samples from which we
    want to estimate CI of Z where Z = samples.mean(). Ideally, we would like to sample from
    the true distribution of Z but that is not possible. We can however sample from empirical
    distributions of X_1, ..., X_G so for each group we are going to do the bootstrap and from
    such dataset estimate the CI.

    Estimating CI through percentiles is discouraged here (https://math.mit.edu/~dav/05.dir/class24-prep-a.pdf).
    Instead it is recommended to estimate bar{x} - mu by estimating bar{x}^* - bar{x} where bar{x}^* is obtained
    through bootstap. Only then we take percentiles.
    """
    if not isinstance(groups, np.ndarray):
        groups = np.array(groups)
    assert ci > 0 and ci < 100, "ci should be a percentage between 0 and 100"
    assert ci > 1, "are you sure you need a CI smaller than 1%? The function expects ci to be a %, careful"

    x_bar = groups.mean()

    G, M = groups.shape
    indices = np.random.randint(0, M, size=(n, G, M))
    j_indices = np.arange(G)[np.newaxis, :, np.newaxis]
    group_samples = groups[j_indices, indices]

    group_samples = group_samples.reshape(n, -1)
    x_bar_star = group_samples.mean(axis=1)
    delta_star = x_bar_star - x_bar
    lower_delta = np.percentile(delta_star, (100 - ci) / 2)
    upper_delta = np.percentile(delta_star, 100 - (100 - ci) / 2)
    return x_bar - upper_delta, x_bar - lower_delta
