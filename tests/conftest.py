# Install jaxtyping import hook before any bergson imports
from jaxtyping import install_import_hook

install_import_hook("bergson.hessians", "beartype.beartype")

import pytest  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402


@pytest.fixture
def model():
    """Randomly initialize a small test model."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    return AutoModelForCausalLM.from_config(config)


@pytest.fixture
def dataset():
    """Create a small test dataset."""
    data = {
        "input_ids": [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ],
        "labels": [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
    }
    return Dataset.from_dict(data)
