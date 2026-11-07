#!/usr/bin/env python3
"""Preregistered H1 clocked two-layer A2A policy for PushT.

This module is self-contained so the fixed official source tree remains read-only.
Every flow connects row ``i`` to row ``i``; endpoints are never reassigned.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import resnet18


ACTION_DIM = 2
HISTORY_STEPS = 8
ACTION_STEPS = 8
LATENT_DIM = 512
HIDDEN_DIM = 512
FLOW_LAYERS = 4
ACTION_HEADS = 8
ACTION_LAYERS = 4
ACTION_MLP_RATIO = 4
EULER_STEPS = 6
CLOCK_CHUNKS = 4
CLOCK_STEPS = 32
KL_WEIGHT = 1e-6
MODEL_SEEDS = (42, 43, 44)
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class H1ModelConfig:
    seed: int = 42
    hierarchical: bool = True
    image_size: int = 224
    latent_dim: int = LATENT_DIM
    hidden_dim: int = HIDDEN_DIM
    flat_hidden_dim: int | None = None
    history_steps: int = HISTORY_STEPS
    action_steps: int = ACTION_STEPS
    clock_chunks: int = CLOCK_CHUNKS
    euler_steps: int = EULER_STEPS
    action_heads: int = ACTION_HEADS
    action_layers: int = ACTION_LAYERS
    flow_layers: int = FLOW_LAYERS

    def __post_init__(self) -> None:
        if self.seed not in MODEL_SEEDS:
            raise ValueError(f"seed must be one of {MODEL_SEEDS}")
        positive = (self.image_size, self.latent_dim, self.hidden_dim)
        if any(value <= 0 for value in positive):
            raise ValueError("model dimensions must be positive")
        fixed = {
            "history_steps": (self.history_steps, HISTORY_STEPS),
            "action_steps": (self.action_steps, ACTION_STEPS),
            "clock_chunks": (self.clock_chunks, CLOCK_CHUNKS),
            "euler_steps": (self.euler_steps, EULER_STEPS),
            "action_heads": (self.action_heads, ACTION_HEADS),
            "action_layers": (self.action_layers, ACTION_LAYERS),
            "flow_layers": (self.flow_layers, FLOW_LAYERS),
        }
        mismatch = [f"{k}={v}, expected={e}" for k, (v, e) in fixed.items() if v != e]
        if mismatch:
            raise ValueError("H1 model contract mismatch: " + "; ".join(mismatch))
        if self.latent_dim % self.action_heads:
            raise ValueError("latent dimension must be divisible by eight heads")
        if self.hierarchical and self.flat_hidden_dim is not None:
            raise ValueError("hierarchical model cannot set flat_hidden_dim")

    @classmethod
    def full(cls, *, seed: int, hierarchical: bool, flat_hidden_dim: int | None = None):
        return cls(seed=seed, hierarchical=hierarchical, flat_hidden_dim=flat_hidden_dim)

    @classmethod
    def mac_preflight(
        cls, *, seed: int = 42, hierarchical: bool, flat_hidden_dim: int | None = None
    ):
        return cls(
            seed=seed,
            hierarchical=hierarchical,
            image_size=64,
            latent_dim=128,
            hidden_dim=128,
            flat_hidden_dim=flat_hidden_dim,
        )


def _replace_batch_norm(module: nn.Module) -> nn.Module:
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = max(1, child.num_features // 16)
            while child.num_features % groups:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            _replace_batch_norm(child)
    return module


class ImageSequenceEncoder(nn.Module):
    """Random ResNet-18 with GroupNorm and ImageNet normalization."""

    def __init__(self, steps: int, latent_dim: int) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = _replace_batch_norm(backbone)
        self.steps = steps
        self.projection = nn.Linear(steps * 512, latent_dim)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[1] != self.steps or images.shape[2] != 3:
            raise ValueError(f"images must be [batch,{self.steps},3,height,width]")
        if images.dtype == torch.uint8:
            images = images.float().div(255.0)
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("images must be finite")
        images = (images - self.mean.to(images)) / self.std.to(images)
        batch = images.shape[0]
        features = self.backbone(images.reshape(-1, *images.shape[2:]))
        return self.projection(features.reshape(batch, -1))


class ConvSequenceEncoder(nn.Module):
    """The official three-layer one-dimensional convolutional entry path."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(3):
            conv = nn.Conv1d(current, hidden_dim, kernel_size=5, stride=2, padding=2)
            nn.init.orthogonal_(conv.weight)
            nn.init.zeros_(conv.bias)
            layers.extend((conv, nn.ReLU()))
            current = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Linear(hidden_dim, latent_dim)
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3 or sequence.shape[1:] != (ACTION_STEPS, ACTION_DIM):
            raise ValueError("sequence must have shape [batch,8,2]")
        if not torch.isfinite(sequence).all():
            raise ValueError("sequence contains non-finite values")
        encoded = self.encoder(sequence.transpose(1, 2)).flatten(1)
        return self.projection(encoded)


class VariationalActionAutoencoder(nn.Module):
    """Trainable 512-D, eight-head, four-layer variational action autoencoder."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.conv_entry = ConvSequenceEncoder(ACTION_DIM, latent_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=ACTION_HEADS,
            dim_feedforward=latent_dim * ACTION_MLP_RATIO,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=ACTION_LAYERS)
        self.encoder_positions = nn.Parameter(torch.zeros(1, ACTION_STEPS, latent_dim))
        self.action_tokens = nn.Linear(ACTION_DIM, latent_dim)
        self.posterior_mean = nn.Linear(latent_dim * 2, latent_dim)
        self.posterior_logvar = nn.Linear(latent_dim * 2, latent_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=ACTION_HEADS,
            dim_feedforward=latent_dim * ACTION_MLP_RATIO,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=ACTION_LAYERS)
        self.decoder_positions = nn.Parameter(torch.zeros(1, ACTION_STEPS, latent_dim))
        self.action_head = nn.Linear(latent_dim, ACTION_DIM)

    def posterior(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        conv = self.conv_entry(actions)
        tokens = self.encoder(self.action_tokens(actions) + self.encoder_positions)
        pooled = tokens.mean(dim=1)
        combined = torch.cat((conv, pooled), dim=-1)
        return self.posterior_mean(combined), self.posterior_logvar(combined).clamp(-20, 10)

    def encode_mean(self, actions: Tensor) -> Tensor:
        return self.posterior(actions)[0]

    def sample(self, actions: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
        mean, logvar = self.posterior(actions)
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        return mean + noise * torch.exp(0.5 * logvar), mean, logvar

    def decode(self, latent: Tensor) -> Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("action latent has an invalid shape")
        tokens = latent[:, None, :].expand(-1, ACTION_STEPS, -1) + self.decoder_positions
        return self.action_head(self.decoder(tokens))


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dimension: int = 64) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("time embedding dimension must be even")
        frequencies = math.pi * 2.0 ** torch.arange(dimension // 2, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, tau: Tensor) -> Tensor:
        angles = tau * self.frequencies.to(tau)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class FlowBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(approximate="tanh"), nn.Linear(width * 4, width)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 3))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, value: Tensor, conditioning: Tensor) -> Tensor:
        gain, scale, shift = self.modulation(conditioning).chunk(3, dim=-1)
        normalized = self.norm(value) * (1 + scale) + shift
        return value + gain * self.mlp(normalized)


class Layer1VelocityField(nn.Module):
    """Official-style four-block conditional velocity field with optional slow state."""

    def __init__(self, latent_dim: int, width: int, *, hierarchical: bool) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.width = width
        self.hierarchical = hierarchical
        condition_parts = 4 if hierarchical else 3
        self.condition_dim = condition_parts * latent_dim
        self.input_projection = nn.Linear(latent_dim, width)
        self.time_embedding = FourierTimeEmbedding(64)
        self.time_projection = nn.Linear(64, width)
        self.condition_projection = nn.Linear(self.condition_dim, width)
        self.blocks = nn.ModuleList(FlowBlock(width) for _ in range(FLOW_LAYERS))
        self.norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, latent_dim)

    def forward(
        self,
        latent: Tensor,
        tau: Tensor,
        head_condition: Tensor,
        goal_condition: Tensor,
        agent_condition: Tensor,
        slow_state: Tensor | None = None,
    ) -> Tensor:
        parts = [head_condition, goal_condition, agent_condition]
        if self.hierarchical:
            if slow_state is None or slow_state.shape != latent.shape:
                raise ValueError("hierarchical velocity requires one slow state per row")
            parts.append(slow_state)
        elif slow_state is not None:
            raise ValueError("flat velocity cannot receive a slow state")
        value = self.input_projection(latent)
        conditioning = self.time_projection(self.time_embedding(tau))
        conditioning = conditioning + self.condition_projection(torch.cat(parts, dim=-1))
        for block in self.blocks:
            value = block(value, conditioning)
        return self.output_projection(self.norm(value))


class ChunkHistoryEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.Linear(CLOCK_CHUNKS * latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, history: Tensor) -> Tensor:
        if history.ndim != 3 or history.shape[1:] != (CLOCK_CHUNKS, self.latent_dim):
            raise ValueError("chunk history must have shape [batch,4,latent_dim]")
        return self.network(history.flatten(1))


class Layer2VelocityField(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.time_embedding = FourierTimeEmbedding(64)
        input_dim = latent_dim * 3 + 64
        modules: list[nn.Module] = []
        dimensions = (input_dim, hidden_dim, hidden_dim, hidden_dim, latent_dim)
        for index, (before, after) in enumerate(zip(dimensions, dimensions[1:])):
            modules.append(nn.Linear(before, after))
            if index < 3:
                modules.append(nn.GELU())
        self.network = nn.Sequential(*modules)

    def forward(
        self, latent: Tensor, tau: Tensor, history_condition: Tensor, goal_condition: Tensor
    ) -> Tensor:
        return self.network(
            torch.cat((latent, history_condition, goal_condition, self.time_embedding(tau)), dim=-1)
        )


class ClockedLayer2(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, euler_steps: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.euler_steps = euler_steps
        self.history_encoder = ChunkHistoryEncoder(latent_dim)
        self.velocity_field = Layer2VelocityField(latent_dim, hidden_dim)

    def flow_loss(
        self, history: Tensor, target: Tensor, goal_condition: Tensor, tau: Tensor
    ) -> Tensor:
        history = history.detach()
        target = target.detach()
        start = history[:, -1]
        mixed = (1 - tau) * start + tau * target
        predicted = self.velocity_field(mixed, tau, self.history_encoder(history), goal_condition)
        return F.mse_loss(predicted, target - start)

    def integrate(self, history: Tensor, goal_condition: Tensor) -> Tensor:
        condition = self.history_encoder(history)
        latent = history[:, -1]
        step_size = 1.0 / self.euler_steps
        for step in range(self.euler_steps):
            tau = torch.full((latent.shape[0], 1), step * step_size, device=latent.device, dtype=latent.dtype)
            latent = latent + step_size * self.velocity_field(latent, tau, condition, goal_condition)
        return latent


def _validate_actions(actions: Tensor, name: str) -> None:
    if actions.ndim != 3 or actions.shape[1:] != (ACTION_STEPS, ACTION_DIM):
        raise ValueError(f"{name} must have shape [batch,8,2]")
    if not torch.isfinite(actions).all():
        raise ValueError(f"{name} must be finite")


class H1Policy(nn.Module):
    def __init__(self, config: H1ModelConfig) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_dim if config.hierarchical else config.flat_hidden_dim
        if width is None:
            width = config.hidden_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.head_encoder = ImageSequenceEncoder(HISTORY_STEPS, config.latent_dim)
            self.goal_encoder = ImageSequenceEncoder(HISTORY_STEPS, config.latent_dim)
            self.agent_encoder = ConvSequenceEncoder(ACTION_DIM, config.latent_dim, config.hidden_dim)
            self.action_autoencoder = VariationalActionAutoencoder(config.latent_dim, config.hidden_dim)
            self.layer1_velocity = Layer1VelocityField(
                config.latent_dim, width, hierarchical=config.hierarchical
            )
            self.layer2 = (
                ClockedLayer2(config.latent_dim, config.hidden_dim, config.euler_steps)
                if config.hierarchical
                else None
            )

    def new_generator(self, device: torch.device | str) -> torch.Generator:
        generator = torch.Generator(device=torch.device(device))
        generator.manual_seed(self.config.seed)
        return generator

    def encode_conditions(
        self, head_cam: Tensor, goal_cam: Tensor, agent_pos: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if agent_pos.ndim != 3 or agent_pos.shape[1:] != (HISTORY_STEPS, ACTION_DIM):
            raise ValueError("agent_pos must have shape [batch,8,2]")
        if not agent_pos.is_floating_point() or not torch.isfinite(agent_pos).all():
            raise ValueError("agent_pos must be finite floating point")
        return self.head_encoder(head_cam), self.goal_encoder(goal_cam), self.agent_encoder(agent_pos)

    def _integrate_layer1(
        self,
        start: Tensor,
        conditions: tuple[Tensor, Tensor, Tensor],
        slow_state: Tensor | None,
    ) -> Tensor:
        latent = start
        step_size = 1.0 / self.config.euler_steps
        for step in range(self.config.euler_steps):
            tau = torch.full((latent.shape[0], 1), step * step_size, device=latent.device, dtype=latent.dtype)
            latent = latent + step_size * self.layer1_velocity(
                latent, tau, *conditions, slow_state
            )
        return latent

    def predict_slow_state(self, action_history: Tensor, goal_condition: Tensor) -> Tensor:
        if self.layer2 is None:
            raise RuntimeError("flat model has no second layer")
        if action_history.ndim != 4 or action_history.shape[1:] != (
            CLOCK_CHUNKS,
            ACTION_STEPS,
            ACTION_DIM,
        ):
            raise ValueError("slow action history must be [batch,4,8,2]")
        batch = action_history.shape[0]
        means = self.action_autoencoder.encode_mean(action_history.reshape(-1, 8, 2))
        history = means.reshape(batch, CLOCK_CHUNKS, self.config.latent_dim).detach()
        return self.layer2.integrate(history, goal_condition)

    def layer1_loss(
        self,
        batch: Mapping[str, Tensor],
        *,
        generator: torch.Generator,
        fixed_tau: Tensor | None = None,
        slow_override: Tensor | None = None,
    ) -> dict[str, Tensor]:
        past = batch["agent_pos"]
        future = batch["future_actions"]
        _validate_actions(future, "future_actions")
        conditions = self.encode_conditions(batch["head_cam"], batch["goal_cam"], past)
        start = self.agent_encoder(past)
        target, mean, logvar = self.action_autoencoder.sample(future, generator)
        if fixed_tau is None:
            tau = torch.rand((future.shape[0], 1), device=future.device, dtype=future.dtype, generator=generator)
        else:
            tau = fixed_tau
        slow_state: Tensor | None = None
        if self.config.hierarchical:
            if slow_override is not None:
                slow_state = slow_override
            elif "slow_history_actions" in batch:
                predicted = self.predict_slow_state(batch["slow_history_actions"], conditions[1])
                mask = batch.get("slow_state_mask", torch.ones((future.shape[0], 1), device=future.device))
                slow_state = predicted * mask.to(predicted)
            else:
                slow_state = torch.zeros_like(start)
        mixed = (1 - tau) * start + tau * target
        velocity = self.layer1_velocity(mixed, tau, *conditions, slow_state)
        flow = F.mse_loss(velocity, target - start)
        integrated = self._integrate_layer1(start, conditions, slow_state)
        latent_consistency = F.mse_loss(integrated, target)
        flow_action = F.l1_loss(self.action_autoencoder.decode(integrated), future)
        teacher_action = F.l1_loss(self.action_autoencoder.decode(target), future)
        kl = -0.5 * torch.mean(1 + logvar - mean.square() - logvar.exp())
        total = flow + latent_consistency + 0.5 * flow_action + 0.5 * teacher_action + KL_WEIGHT * kl
        return {
            "total": total,
            "flow": flow,
            "latent_consistency": latent_consistency,
            "flow_action_l1": flow_action,
            "teacher_action_l1": teacher_action,
            "kl": kl,
        }

    def layer2_loss(
        self, batch: Mapping[str, Tensor], *, generator: torch.Generator
    ) -> Tensor:
        if self.layer2 is None:
            return next(self.parameters()).new_zeros(())
        history_actions = batch["history_actions"]
        target_actions = batch["target_actions"]
        batch_size = history_actions.shape[0]
        history = self.action_autoencoder.encode_mean(history_actions.reshape(-1, 8, 2))
        history = history.reshape(batch_size, CLOCK_CHUNKS, self.config.latent_dim).detach()
        target = self.action_autoencoder.encode_mean(target_actions).detach()
        goal_condition = self.goal_encoder(batch["goal_cam"])
        tau = torch.rand((batch_size, 1), device=target.device, dtype=target.dtype, generator=generator)
        return self.layer2.flow_loss(history, target, goal_condition, tau)

    @torch.no_grad()
    def predict_actions(
        self,
        head_cam: Tensor,
        goal_cam: Tensor,
        agent_pos: Tensor,
        slow_state: Tensor | None = None,
    ) -> Tensor:
        conditions = self.encode_conditions(head_cam, goal_cam, agent_pos)
        start = self.agent_encoder(agent_pos)
        if self.config.hierarchical and slow_state is None:
            slow_state = torch.zeros_like(start)
        latent = self._integrate_layer1(start, conditions, slow_state)
        return self.action_autoencoder.decode(latent)


def parameter_manifest(model: nn.Module) -> dict[str, Any]:
    parameters = {
        name: {
            "shape": list(parameter.shape),
            "count": parameter.numel(),
            "trainable": parameter.requires_grad,
        }
        for name, parameter in model.named_parameters()
    }
    return {
        "parameters": parameters,
        "total": sum(item["count"] for item in parameters.values()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def layer1_velocity_parameter_count(latent_dim: int, width: int, condition_dim: int) -> int:
    """Closed-form count for Layer1VelocityField; checked by the contract tests."""
    return 44 * width * width + (2 * latent_dim + 64 + condition_dim + 37) * width + latent_dim


def select_flat_width(
    hierarchical_model: H1Policy, *, maximum_width: int = 8192
) -> dict[str, int | float]:
    """Lexicographically scan positive widths and select the closest flat capacity."""
    if not hierarchical_model.config.hierarchical:
        raise ValueError("capacity target must be hierarchical")
    manifest = parameter_manifest(hierarchical_model)
    target = int(manifest["trainable"])
    hierarchical_velocity = sum(p.numel() for p in hierarchical_model.layer1_velocity.parameters())
    layer2_count = sum(p.numel() for p in hierarchical_model.layer2.parameters())
    shared = target - hierarchical_velocity - layer2_count
    condition_dim = 3 * hierarchical_model.config.latent_dim
    candidates = []
    for width in range(1, maximum_width + 1):
        count = shared + layer1_velocity_parameter_count(
            hierarchical_model.config.latent_dim, width, condition_dim
        )
        candidates.append((abs(count - target), width, count))
    difference, width, count = min(candidates)
    relative = difference / target
    if relative > 0.001:
        raise RuntimeError(
            f"h1_infrastructure_blocked: closest flat capacity differs by {relative:.8f}"
        )
    return {
        "width": width,
        "hierarchical_trainable": target,
        "flat_trainable": count,
        "absolute_difference": difference,
        "relative_difference": relative,
    }


def build_capacity_matched_pair(config: H1ModelConfig) -> tuple[H1Policy, H1Policy, dict[str, Any]]:
    if not config.hierarchical:
        raise ValueError("pair construction requires a hierarchical config")
    hierarchical = H1Policy(config)
    selection = select_flat_width(hierarchical)
    flat_config = H1ModelConfig(
        **{
            **asdict(config),
            "hierarchical": False,
            "flat_hidden_dim": int(selection["width"]),
        }
    )
    flat = H1Policy(flat_config)
    observed = parameter_manifest(flat)["trainable"]
    if observed != selection["flat_trainable"]:
        raise RuntimeError("h1_infrastructure_blocked: analytical flat count mismatch")
    ledger = {
        "selection": selection,
        "hierarchical": parameter_manifest(hierarchical),
        "flat": parameter_manifest(flat),
    }
    return hierarchical, flat, ledger


def save_checkpoint(path: Path, model: H1Policy, *, update: int, ema_state: Mapping[str, Tensor]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "config": asdict(model.config),
            "update": int(update),
            "ema_state": dict(ema_state),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[H1Policy, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported H1 checkpoint")
    model = H1Policy(H1ModelConfig(**payload["config"]))
    model.load_state_dict(payload["ema_state"], strict=True)
    return model, {"update": int(payload["update"]), "config": payload["config"]}
