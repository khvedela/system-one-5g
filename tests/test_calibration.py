import torch

from system_one.calibration import fit_temperature


def test_temperature_improves_overconfident_nll_without_changing_logits_or_choices():
    torch.set_num_threads(1)
    logits = torch.tensor([[8., 0, 0, 0, 0]] * 100)
    original = logits.clone()
    labels = torch.tensor([0] * 80 + [1] * 20)
    temperature = fit_temperature(logits, labels)
    assert temperature > 1
    loss = torch.nn.functional.cross_entropy
    assert loss(logits / temperature, labels) < loss(logits, labels)
    torch.testing.assert_close(logits, original)
    assert torch.equal(logits.argmax(dim=1), (logits / temperature).argmax(dim=1))
