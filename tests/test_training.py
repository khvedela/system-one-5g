import numpy as np
import torch

from system_one.training import TrainingConfig, train, checkpoint_logits


def test_training_is_reproducible_uses_only_train_normalization_and_reloads(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 6))
    y = (x[:, 0] > 0).astype(np.int64)
    config = TrainingConfig(epochs=2, hidden_sizes=(8, 4), batch_size=20)
    first, history = train(x[:60], y[:60], x[60:] + 100, y[60:], config)
    second, repeat_history = train(x[:60], y[:60], x[60:] + 100, y[60:], config)
    np.testing.assert_allclose(first["mean"], x[:60].mean(axis=0))
    assert history == repeat_history
    for key in first["state_dict"]:
        torch.testing.assert_close(first["state_dict"][key], second["state_dict"][key], rtol=0, atol=0)
    path = tmp_path / "model.pt"
    torch.save(first, path)
    loaded = torch.load(path, weights_only=True)
    torch.testing.assert_close(checkpoint_logits(first, x), checkpoint_logits(loaded, x))
