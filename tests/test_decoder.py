import pytest
import torch
from decoder import Decoder


@pytest.fixture
def decoder_config() -> dict[str, int]:
    return {"C": 3, "W": 64, "H": 64, "stoch_dim": 32, "deter_dim": 64, "hidden_dim": 48}


@pytest.fixture
def decoder(decoder_config: dict[str, int]) -> Decoder:
    return Decoder(**decoder_config)


@pytest.mark.parametrize("batch_size,time_steps", [(1, 1), (2, 3)])
def test_decoder_forward_shape(
    decoder: Decoder, decoder_config: dict[str, int], batch_size: int, time_steps: int
) -> None:
    stoch = torch.randn(batch_size, time_steps, decoder_config["stoch_dim"])
    deter = torch.randn(batch_size, time_steps, decoder_config["deter_dim"])
    recon = decoder(stoch, deter)
    assert recon.shape == (batch_size, time_steps, decoder_config["C"], decoder_config["W"], decoder_config["H"])
