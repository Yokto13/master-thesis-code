import numpy as np


def generate_random_logits(batch_shape, n, dtype=np.float64):
    """Generate random logits from a normal distribution."""
    return np.random.randn(*batch_shape, n).astype(dtype)


def generate_uniform_logits(batch_shape, n, dtype=np.float64):
    """Generate uniform logits (all equal, resulting in uniform probabilities)."""
    return np.ones((*batch_shape, n), dtype=dtype)


def generate_one_hot_logits(batch_shape, n, dtype=np.float64):
    """Generate one-hot logits (single hot index per batch)."""
    logits = np.full((*batch_shape, n), -1000.0, dtype=dtype)  # Very negative values

    # For each batch element, randomly select one index to be hot
    for i in range(np.prod(batch_shape)):
        flat_idx = np.unravel_index(i, batch_shape)
        hot_idx = np.random.randint(0, n)
        logits[flat_idx][hot_idx] = 0.0  # Set one index to 0 (much higher than -1000)

    return logits


def test_splitting_logic(logits, logit_type="unknown", dtype=None):
    """Test the two-hot splitting logic with given logits."""
    if dtype is None:
        dtype = logits.dtype

    n = logits.shape[-1]
    m = (n - 1) // 2
    batch_shape = logits.shape[:-1]

    # Convert logits to probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    # Create buckets
    x = np.linspace(-20, 20, n).astype(dtype)
    buckets = np.sign(x) * (np.exp(np.abs(x)) - 1)

    # Naive weighted average
    naive_wavg = np.sum(probs * buckets, axis=-1)

    # Split probabilities and buckets
    p1 = probs[..., :m]
    p2 = probs[..., m : m + 1]
    p3 = probs[..., m + 1 :]

    b1 = buckets[..., :m]
    b2 = buckets[..., m : m + 1]
    b3 = buckets[..., m + 1 :]

    # Elaborate weighted average using splitting logic
    elaborate_wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)

    # Calculate difference
    diff = np.abs(naive_wavg - elaborate_wavg)

    print(f"--- Testing {logit_type} logits ({dtype.__name__}) ---")
    print(f"Batch shape: {batch_shape}, N buckets: {n}")
    print(f"Max difference: {np.max(diff):.8f}")
    print(f"Mean difference: {np.mean(diff):.8f}")
    print(f"Logic equivalent: {np.allclose(naive_wavg, elaborate_wavg, atol=1e-4)}")

    # Additional diagnostics for debugging
    if np.max(diff) > 1e-4:
        print("WARNING: Large differences detected!")
        worst_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(
            f"Worst case at idx {worst_idx}: naive={naive_wavg[worst_idx]:.8f} dreamer={elaborate_wavg[worst_idx]:.8f}"
        )

    print()
    return np.allclose(naive_wavg, elaborate_wavg, atol=1e-4)


def run_comprehensive_test(dtype=np.float64):
    """Run tests with all logit types for a given dtype."""
    n = 255
    batch_shape = (64, 32)  # Smaller batch for faster testing

    print(f"=== Comprehensive Test for {dtype.__name__} ===")

    # Test 1: Random logits
    random_logits = generate_random_logits(batch_shape, n, dtype)
    result1 = test_splitting_logic(random_logits, "random", dtype)

    # Test 2: Uniform logits
    uniform_logits = generate_uniform_logits(batch_shape, n, dtype)
    result2 = test_splitting_logic(uniform_logits, "uniform", dtype)

    # Test 3: One-hot logits
    one_hot_logits = generate_one_hot_logits(batch_shape, n, dtype)
    result3 = test_splitting_logic(one_hot_logits, "one-hot", dtype)

    all_passed = result1 and result2 and result3
    print(f"All tests passed for {dtype.__name__}: {all_passed}")
    print("=" * 50)

    return all_passed


if __name__ == "__main__":
    # Test with different data types
    dtypes = [np.float32, np.float64, np.float16]

    all_results = []
    for dtype in dtypes:
        result = run_comprehensive_test(dtype)
        all_results.append(result)

    print("\n=== SUMMARY ===")
    for dtype, result in zip(dtypes, all_results):
        status = "PASS" if result else "FAIL"
        print(f"{dtype.__name__}: {status}")

    overall_success = all(all_results)
    print(f"\nOverall test result: {'SUCCESS' if overall_success else 'FAILURE'}")
