import numpy as np


def test_blocks_preserve_alignment_seed_and_gaps():
    from tails.resampling import block_bootstrap_indices, resample_aligned

    valid = np.array([True, True, True, False, True, True])
    idx = block_bootstrap_indices(valid, mean_block_length=3, seed=11)
    assert idx[3] == -1
    assert np.all(valid[idx[valid]])
    np.testing.assert_array_equal(idx, block_bootstrap_indices(valid, mean_block_length=3, seed=11))
    x = np.vstack([np.arange(6.), 2*np.arange(6.)])
    x[:, 3] = np.nan
    draw = resample_aligned(x, mean_block_length=3, seed=11)
    np.testing.assert_allclose(draw[1], 2*draw[0], equal_nan=True)
    assert np.isnan(draw[:, 3]).all()


def test_bootstrap_reports_failures_instead_of_dropping_them():
    from tails.resampling import bootstrap_statistic

    calls = []

    def statistic(x):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("unusable fit")
        return float(x.mean())

    out = bootstrap_statistic(np.arange(30.)[None, :], statistic, n_boot=8, mean_block_length=3, seed=7)
    assert out.diagnostics.n_attempted == 8
    assert out.diagnostics.n_successful == 7
    assert np.isnan(out.lower) and np.isnan(out.upper)
    assert out.draws.shape == (8,)
    assert sum(out.diagnostics.failures.values()) == 1

