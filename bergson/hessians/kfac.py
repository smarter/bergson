import os
from dataclasses import dataclass

import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch import Tensor

from bergson.collector.collector import HookCollectorBase
from bergson.hessians.sharded_computation import ShardedMul
from bergson.utils.utils import assert_type


@dataclass(kw_only=True)
class CovarianceCollector(HookCollectorBase):
    """
    Collects activation and gradient covariances for EKFAC.

    Computes:
        A_cov = sum over batches of (X^T @ X)  for activations
        S_cov = sum over batches of (G^T @ G)  for gradients

    where X is input activations [N*S, I] and G is output gradients [N*S, O].

    Covariances are computed in float32 to avoid numerical issues with bfloat16.

    Note: To be compatible with gradient checkpointing, activations are saved
    in forward_hook and covariances are computed in backward_hook.
    """

    path: str

    def setup(self) -> None:
        """Initialize covariance storage dictionaries."""
        self.A_cov_dict = {}
        self.S_cov_dict = {}
        self.activation_cache = {}  # Cache activations for use in backward
        self.shard_computer = ShardedMul()
        # Initialize sharded covariance matrices for ALL modules in target_info
        self.shard_computer._init_covariance_dict(
            activation_covariance_dict=self.A_cov_dict,
            gradient_covariance_dict=self.S_cov_dict,
            target_info=self.target_info,
        )

    def forward_hook(self, module: nn.Module, a: Tensor) -> None:
        """Save activations for computing covariance in backward pass."""
        name = assert_type(str, module._name)
        mask = self._current_valid_mask
        assert mask is not None, "Valid mask not set for forward hook."

        # a: [N, S, I], valid_masks: [N, S] -> select valid positions
        # Save valid activations for use in backward_hook
        # This pattern ensures gradient checkpointing doesn't double-count
        a_bi = a[mask]  # [num_valid, I]
        self.activation_cache[name] = a_bi

    def backward_hook(self, module: nn.Module, g: Tensor) -> None:
        """Compute both activation and gradient covariances using cached activations."""
        name = assert_type(str, module._name)

        # Retrieve cached activations
        a_bi = self.activation_cache.get(name)
        if a_bi is None:
            return  # Skip if no cached activations

        # Convert to float32 to avoid bfloat16 numerical issues
        a_bi = a_bi.float()

        # Compute activation covariance: A^T @ A
        A_cov_ki = self.A_cov_dict[name]
        local_update_aa = a_bi.mT @ a_bi

        # All-reduce across ranks
        if dist.is_initialized():
            dist.all_reduce(local_update_aa, op=dist.ReduceOp.SUM)

        # Extract our shard and accumulate
        start_row = self.rank * A_cov_ki.shape[0]
        end_row = (self.rank + 1) * A_cov_ki.shape[0]
        A_cov_ki.add_(local_update_aa[start_row:end_row, :])

        # Compute gradient covariance: G^T @ G
        S_cov_po = self.S_cov_dict[name]
        mask = self._current_valid_mask

        # g: [N, S, O], mask: [N, S] -> select valid positions
        # Convert to float32 to avoid bfloat16 numerical issues
        g_bo = g[mask].float()  # [num_valid, O]

        # Compute local covariance
        local_update_gg = g_bo.mT @ g_bo

        # All-reduce across ranks
        if dist.is_initialized():
            dist.all_reduce(local_update_gg, op=dist.ReduceOp.SUM)

        # Extract our shard and accumulate
        start_row = self.rank * S_cov_po.shape[0]
        end_row = (self.rank + 1) * S_cov_po.shape[0]
        S_cov_po.add_(local_update_gg[start_row:end_row, :])

        # Clear cache to free memory
        del self.activation_cache[name]

    def process_batch(self, indices: list[int], **kwargs) -> None:
        """No per-batch processing needed for covariance collection."""
        pass

    def teardown(self) -> None:
        """Save covariance matrices to disk."""
        activation_path = os.path.join(self.path, "activation_sharded")
        gradient_path = os.path.join(self.path, "gradient_sharded")

        os.makedirs(activation_path, exist_ok=True)
        os.makedirs(gradient_path, exist_ok=True)
        self.logger.info(
            f"Saving sharded covariance matrices to {activation_path} "
            f"and {gradient_path}"
        )
        # Save sharded covariance matrices
        save_file(
            self.A_cov_dict,
            os.path.join(activation_path, f"shard_{self.rank}.safetensors"),
        )
        save_file(
            self.S_cov_dict,
            os.path.join(gradient_path, f"shard_{self.rank}.safetensors"),
        )
