"""Ground truth collector for EKFAC testing."""

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float, Float32
from torch import Tensor

from bergson.collector.collector import HookCollectorBase
from bergson.utils.utils import assert_type


@dataclass(kw_only=True)
class GroundTruthCovarianceCollector(HookCollectorBase):
    """Collects activation and gradient covariances in float32."""

    activation_covariances: MutableMapping[str, Float32[Tensor, "i i"]]
    gradient_covariances: MutableMapping[str, Float32[Tensor, "o o"]]

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def forward_hook(self, module: nn.Module, a: Float[Tensor, "n s i"]) -> None:
        name = assert_type(str, module._name)
        mask = self._current_valid_mask

        # Select valid positions and convert to float32
        if mask is not None:
            a_bi: Float32[Tensor, "b i"] = a[mask].float()
        else:
            a_bi = a.reshape(-1, a.shape[-1]).float()

        update_ii: Float32[Tensor, "i i"] = a_bi.mT @ a_bi

        if name not in self.activation_covariances:
            self.activation_covariances[name] = update_ii
        else:
            self.activation_covariances[name].add_(update_ii)

    def backward_hook(self, module: nn.Module, g: Float[Tensor, "n s o"]) -> None:
        name = assert_type(str, module._name)
        mask = self._current_valid_mask

        # Select valid positions and convert to float32
        if mask is not None:
            g_bo: Float32[Tensor, "b o"] = g[mask].float()
        else:
            g_bo = g.reshape(-1, g.shape[-1]).float()

        update_oo: Float32[Tensor, "o o"] = g_bo.mT @ g_bo

        if name not in self.gradient_covariances:
            self.gradient_covariances[name] = update_oo
        else:
            self.gradient_covariances[name].add_(update_oo)

    def process_batch(self, indices: list[int], **kwargs) -> None:
        pass


@dataclass(kw_only=True)
class GroundTruthNonAmortizedLambdaCollector(HookCollectorBase):
    """Computes eigenvalue corrections using non-amortized method (for reference)."""

    eigenvalue_corrections: MutableMapping[str, Float32[Tensor, "o i"]]
    eigenvectors_activations: Mapping[str, Float32[Tensor, "i i"]]
    eigenvectors_gradients: Mapping[str, Float32[Tensor, "o o"]]
    device: torch.device

    def setup(self) -> None:
        self.activation_cache: dict[str, Float[Tensor, "n s i"]] = {}

    def teardown(self) -> None:
        self.activation_cache.clear()

    def forward_hook(self, module: nn.Module, a: Float[Tensor, "n s i"]) -> None:
        name = assert_type(str, module._name)
        self.activation_cache[name] = a

    def backward_hook(self, module: nn.Module, g: Float[Tensor, "n s o"]) -> None:
        name = assert_type(str, module._name)
        Q_a: Float32[Tensor, "i j"] = self.eigenvectors_activations[name].to(
            device=self.device
        )
        Q_g: Float32[Tensor, "o p"] = self.eigenvectors_gradients[name].to(
            device=self.device
        )

        a_nsi: Float32[Tensor, "n s i"] = self.activation_cache[name].float()
        g_nso: Float32[Tensor, "n s o"] = g.float()

        # dW = g ⊗ a (per-position weight gradient)
        dW_nsoi: Float32[Tensor, "n s o i"] = torch.einsum(
            "n s o, n s i -> n s o i", g_nso, a_nsi
        )

        # dW @ Q_a (transform by activation eigenvectors)
        dW_Qa_nsoj: Float32[Tensor, "n s o j"] = torch.einsum(
            "n s o i, i j -> n s o j", dW_nsoi, Q_a
        )

        # Q_g^T @ dW @ Q_a (fully transformed, in eigenbasis)
        Qg_dW_Qa_nspj: Float32[Tensor, "n s p j"] = torch.einsum(
            "o p, n s o j -> n s p j", Q_g, dW_Qa_nsoj
        )

        # Sum over sequence, square, sum over batch -> λ = E[(Q_g^T dW Q_a)²]
        Qg_dW_Qa_npj: Float32[Tensor, "n p j"] = Qg_dW_Qa_nspj.sum(dim=1)
        lambda_pj: Float32[Tensor, "p j"] = (Qg_dW_Qa_npj**2).sum(dim=0)

        if name not in self.eigenvalue_corrections:
            self.eigenvalue_corrections[name] = lambda_pj
        else:
            self.eigenvalue_corrections[name].add_(lambda_pj)

    def process_batch(self, indices: list[int], **kwargs) -> None:
        pass


@dataclass(kw_only=True)
class GroundTruthAmortizedLambdaCollector(HookCollectorBase):
    """Computes eigenvalue corrections using amortized method (matches EKFAC)."""

    eigenvalue_corrections: MutableMapping[str, Float32[Tensor, "o i"]]
    eigenvectors_activations: Mapping[str, Float32[Tensor, "i i"]]
    eigenvectors_gradients: Mapping[str, Float32[Tensor, "o o"]]
    device: torch.device

    def setup(self) -> None:
        self.activation_cache: dict[str, Float[Tensor, "n s i"]] = {}

    def teardown(self) -> None:
        self.activation_cache.clear()

    def forward_hook(self, module: nn.Module, a: Float[Tensor, "n s i"]) -> None:
        name = assert_type(str, module._name)
        self.activation_cache[name] = a

    def backward_hook(self, module: nn.Module, g: Float[Tensor, "n s o"]) -> None:
        name = assert_type(str, module._name)
        Q_a: Float32[Tensor, "i j"] = self.eigenvectors_activations[name].to(
            device=self.device
        )
        Q_g: Float32[Tensor, "o p"] = self.eigenvectors_gradients[name].to(
            device=self.device
        )

        a_nsi: Float32[Tensor, "n s i"] = self.activation_cache[name].float()
        g_nso: Float32[Tensor, "n s o"] = g.float()

        # a @ Q_a (transform activations by eigenvectors)
        a_Qa_nsj: Float32[Tensor, "n s j"] = torch.einsum(
            "n s i, i j -> n s j", a_nsi, Q_a
        )

        # Q_g^T @ g (transform gradients by eigenvectors)
        Qg_g_nsp: Float32[Tensor, "n s p"] = torch.einsum(
            "o p, n s o -> n s p", Q_g, g_nso
        )

        # λ = E[(Q_g^T g ⊗ a Q_a)²] = E[(Q_g^T dW Q_a)²]
        # Outer product and sum over sequence gives per-sample transformed gradient
        Qg_dW_Qa_npj: Float32[Tensor, "n p j"] = torch.einsum(
            "n s p, n s j -> n p j", Qg_g_nsp, a_Qa_nsj
        )
        lambda_pj: Float32[Tensor, "p j"] = (Qg_dW_Qa_npj**2).sum(dim=0).contiguous()

        if name not in self.eigenvalue_corrections:
            self.eigenvalue_corrections[name] = lambda_pj
        else:
            self.eigenvalue_corrections[name].add_(lambda_pj)

    def process_batch(self, indices: list[int], **kwargs) -> None:
        pass
