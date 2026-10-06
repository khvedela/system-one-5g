import numpy as np

from system_one.data import DatasetConfig, add_noise, generate_clean, label_states, save_dataset, state_from_row
from system_one.state import RANGES, FEATURE_NAMES
from system_one.data import split_indices


def test_clean_generation_is_seeded_valid_and_covers_scenarios():
    config = DatasetConfig(samples=1000, seed=7)
    states, scenarios = generate_clean(config)
    other, other_scenarios = generate_clean(config)
    np.testing.assert_array_equal(states, other)
    np.testing.assert_array_equal(scenarios, other_scenarios)
    assert set(scenarios) == set(range(5))
    assert not np.array_equal(states, generate_clean(DatasetConfig(samples=1000, seed=8))[0])
    for row in states:
        state_from_row(row)
    for j, name in enumerate(FEATURE_NAMES):
        lower, upper = RANGES[name]
        assert np.all((states[:, j] >= lower) & (states[:, j] <= upper))


def test_labeling_and_csv_export(tmp_path):
    config = DatasetConfig(samples=1000)
    states, scenarios = generate_clean(config)
    labels = label_states(states)
    assert set(labels) == set(range(5))
    destination = tmp_path / "dataset"
    save_dataset(destination, states, states, labels, scenarios, config)
    assert len((destination / "states.csv").read_text().splitlines()) == 1001
    assert '"policy_version": "server-policy-v1"' in (destination / "metadata.json").read_text()


def test_noise_is_seeded_keeps_labels_latent_and_does_not_mutate_clean():
    config = DatasetConfig(samples=1000)
    clean, _ = generate_clean(config)
    original = clean.copy()
    labels = label_states(clean)
    observed = add_noise(clean, config)
    np.testing.assert_array_equal(clean, original)
    np.testing.assert_array_equal(labels, label_states(clean))
    np.testing.assert_array_equal(observed, add_noise(clean, config))
    np.testing.assert_array_equal(observed[:, [1, 5]], clean[:, [1, 5]])
    assert not np.array_equal(observed, clean)
    for row in observed:
        state_from_row(row)


def test_zero_noise_is_identity():
    config = DatasetConfig(samples=100, cpu_noise_std=0, latency_noise_std=0,
                           request_rate_noise_std=0, error_rate_noise_std=0)
    clean, _ = generate_clean(config)
    np.testing.assert_array_equal(clean, add_noise(clean, config))


def test_splits_are_disjoint_exhaustive_and_reproducible():
    clean, _ = generate_clean(DatasetConfig(samples=1000))
    labels = label_states(clean)
    splits = split_indices(labels, 42)
    repeat = split_indices(labels, 42)
    for name, indices in splits.items():
        np.testing.assert_array_equal(indices, repeat[name])
        assert set(labels[indices]) == set(range(5))
    primary = [set(splits[k]) for k in ("train", "validation", "test")]
    assert [len(s) for s in primary] == [700, 150, 150]
    assert not (primary[0] & primary[1] or primary[0] & primary[2] or primary[1] & primary[2])
    assert set.union(*primary) == set(range(1000))
    assert not set(splits["selection"]) & set(splits["calibration"])
    assert set(splits["selection"]) | set(splits["calibration"]) == primary[1]


def test_expanded_dataset_is_versioned_seeded_and_keeps_the_policy(tmp_path):
    config = DatasetConfig(samples=1000, seed=142, broad_fraction=.4, boundary_fraction=.2)
    clean, scenarios = generate_clean(config)
    np.testing.assert_array_equal(clean, generate_clean(config)[0])
    assert (scenarios == 5).sum() == 400
    assert (scenarios == 6).sum() == 200
    save_dataset(tmp_path / "v2", add_noise(clean, config), clean, label_states(clean), scenarios, config)
    metadata = (tmp_path / "v2/metadata.json").read_text()
    assert '"dataset_version": "server-data-v2"' in metadata
    assert '"policy_version": "server-policy-v1"' in metadata
