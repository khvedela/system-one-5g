import numpy as np

from system_one.uncertainty import uncertainty_metrics, select_threshold


def test_threshold_reduces_accepted_errors_without_rewriting_probabilities():
    labels = np.array([0, 1, 4])
    probabilities = np.array([[.9, .025, .025, .025, .025],
                              [.6, .1, .1, .1, .1], [.025, .025, .025, .025, .9]])
    original = probabilities.copy()
    low = uncertainty_metrics(labels, probabilities, .5)
    high = uncertainty_metrics(labels, probabilities, .8)
    assert low["accepted_accuracy"] == .5
    assert high["accepted_accuracy"] == 1
    assert high["coverage"] == 1 / 3
    assert high["forced_escalation_rate"] == 1 / 3
    threshold, _, met = select_threshold(labels, probabilities, min_accepted=1, thresholds=(.5, .8))
    assert met and threshold == .8
    np.testing.assert_array_equal(probabilities, original)


def test_empty_coverage_has_no_fake_accuracy_and_unmet_target_is_explicit():
    result = uncertainty_metrics(np.array([0]), np.array([[.2] * 5]), .9)
    assert result["accepted_accuracy"] is None
    threshold, _, met = select_threshold(np.array([1]), np.array([[.9, .025, .025, .025, .025]]),
                                         min_accepted=1, thresholds=(.5, .8))
    assert not met and threshold == .8
