import numpy as np
import pytest
from bootstrap import bootstrapped_CI


class TestBootstrappedCI:
    def test_returns_two_values(self):
        """Output should be a pair (lower, upper)."""
        data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        result = bootstrapped_CI(data, n=2, ci=95)
        assert len(result) == 2

    def test_lower_le_upper(self):
        """Lower bound must not exceed upper bound."""
        data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        lo, hi = bootstrapped_CI(data, n=2, ci=95)
        assert lo <= hi

    def test_mean_within_ci(self):
        """Sample mean should lie within the CI."""
        data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        lo, hi = bootstrapped_CI(data, n=1000, ci=95)
        assert lo <= np.mean(data) <= hi

    def test_constant_data_tight_ci(self):
        """Constant data has zero variance; CI should collapse to a single point."""
        data = [[5.0] * 20]
        lo, hi = bootstrapped_CI(data, n=10, ci=95)
        assert lo == pytest.approx(5.0)
        assert hi == pytest.approx(5.0)

    def test_bounds_within_data_range(self):
        """Bootstrap means cannot exceed the observed data range."""
        data = [[2.0, 4.0, 6.0, 8.0, 10.0]]
        lo, hi = bootstrapped_CI(data, n=3, ci=95)
        assert lo >= min(data[0])
        assert hi <= max(data[0])

    def test_single_element(self):
        """Single-element data: CI should be the element itself."""
        data = [[7.0]]
        lo, hi = bootstrapped_CI(data, n=3, ci=95)
        assert lo == pytest.approx(7.0)
        assert hi == pytest.approx(7.0)

    def test_numpy_array_input(self):
        """Should accept a numpy array, not just a list."""
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        lo, hi = bootstrapped_CI(data, n=3, ci=95)
        assert lo <= hi

    def test_groups(self):
        """Should handle multiple groups of data."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        lo, hi = bootstrapped_CI(data, n=3, ci=95)
        assert lo <= hi
