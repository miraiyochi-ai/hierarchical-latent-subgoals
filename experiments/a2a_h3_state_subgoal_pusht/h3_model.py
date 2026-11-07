#!/usr/bin/env python3
"""H3 state-subgoal hierarchy with the frozen official LeWM observation path."""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

H1_DIR = Path(__file__).resolve().parents[1] / "a2a_h1_clockwork_pusht_gate"
if str(H1_DIR) not in sys.path: sys.path.insert(0, str(H1_DIR))
from h1_model import (  # noqa: E402
    ACTION_DIM, ACTION_STEPS, FLOW_LAYERS, HISTORY_STEPS, KL_WEIGHT, MODEL_SEEDS,
    ConvSequenceEncoder, FlowBlock, FourierTimeEmbedding, ImageSequenceEncoder,
    VariationalActionAutoencoder, layer1_velocity_parameter_count, parameter_manifest,
)

EXPERIMENT_ID = "a2a_h3_state_subgoal_pusht"
CHECKPOINT_VERSION = 3
S2_DIM = 192
LATENT_DIM = 512
HIDDEN_DIM = 512
EULER_STEPS = 6
L2_WINDOW = 16
LEWM_IMAGE_SIZE = 224
# Checkpoint identity is enforced by the SHA-256 check (LEWM_SHA256) in stage G0, so only the
# location is configurable. Defaults point at third_party/hi-lewm/ (see README); override with
# H3_LEWM_CODE / H3_LEWM_CHECKPOINT.
_REPO_ROOT = Path(__file__).resolve().parents[2]
LEWM_CODE = Path(os.environ.get("H3_LEWM_CODE", _REPO_ROOT / "third_party/hi-lewm/code"))
LEWM_CHECKPOINT = Path(os.environ.get("H3_LEWM_CHECKPOINT", _REPO_ROOT / "third_party/hi-lewm/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt"))
LEWM_SHA256 = "b87805747d40037841877ce7b99b7dda3ebe7a52202c0ba46bf0006ab5d6f008"


@dataclass(frozen=True)
class H3ModelConfig:
    seed: int = 42
    hierarchical: bool = True
    flat_hidden_dim: int | None = None
    image_size: int = 224
    latent_dim: int = LATENT_DIM
    hidden_dim: int = HIDDEN_DIM
    s2_dim: int = S2_DIM
    euler_steps: int = EULER_STEPS

    def __post_init__(self) -> None:
        if self.seed not in MODEL_SEEDS: raise ValueError("seed must be 42, 43, or 44")
        if (self.image_size, self.latent_dim, self.hidden_dim, self.s2_dim, self.euler_steps) != (224, 512, 512, 192, 6):
            raise ValueError("H3 dimensions are preregistered and cannot be changed")
        if self.hierarchical != (self.flat_hidden_dim is None):
            raise ValueError("flat_hidden_dim is required only for the flat model")


def _register_lewm_aliases(code_root: Path = LEWM_CODE) -> None:
    if str(code_root) not in sys.path: sys.path.insert(0, str(code_root))
    import h_le_wm.baseline.adapter as baseline_adapter
    import h_le_wm.models.jepa as jepa
    import h_le_wm.models.latent_action as latent_action
    import h_le_wm.models.vq as vq
    import h_le_wm.models.waypoint_sampling as waypoint_sampling
    _ = baseline_adapter.ARPredictor
    aliases = {"hi_jepa": jepa, "hi_module": latent_action, "hi_vq": vq, "hi_waypoint_sampling": waypoint_sampling, "module": sys.modules.get("_baseline_lewm_module")}
    if aliases["module"] is None: raise RuntimeError("h3_input_blocked: missing LeWM compatibility module")
    for name, module in aliases.items(): sys.modules.setdefault(name, module)


def official_preprocess(images: Tensor) -> Tensor:
    """Official order: ToImage -> float/255 -> ImageNet normalize -> resize 224."""
    if images.ndim not in (4, 5) or images.shape[-3] != 3:
        raise ValueError("LeWM images must be BCHW or BTCHW")
    shape = images.shape
    flat = images.reshape(-1, *shape[-3:])
    if flat.dtype == torch.uint8: flat = flat.float().div(255.0)
    elif not flat.is_floating_point(): raise ValueError("images must be uint8 or floating point")
    mean = torch.tensor((0.485, 0.456, 0.406), device=flat.device, dtype=flat.dtype)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=flat.device, dtype=flat.dtype)[None, :, None, None]
    flat = (flat - mean) / std
    if flat.shape[-2:] != (LEWM_IMAGE_SIZE, LEWM_IMAGE_SIZE):
        flat = F.interpolate(flat, size=(LEWM_IMAGE_SIZE, LEWM_IMAGE_SIZE), mode="bilinear", align_corners=False)
    return flat.reshape(*shape[:-3], 3, LEWM_IMAGE_SIZE, LEWM_IMAGE_SIZE)


class FrozenLeWMEncoder(nn.Module):
    def __init__(self, checkpoint: Path = LEWM_CHECKPOINT, code_root: Path = LEWM_CODE) -> None:
        super().__init__()
        _register_lewm_aliases(code_root)
        model = torch.load(checkpoint, weights_only=False, map_location="cpu")
        self.encoder, self.projector = model.encoder, model.projector
        self.requires_grad_(False); self.eval()
        if self.output_dim() != S2_DIM: raise RuntimeError("h3_input_blocked: LeWM projector output is not 192")

    def train(self, mode: bool = True):
        super().train(False); self.encoder.eval(); self.projector.eval(); return self

    def output_dim(self) -> int:
        for module in reversed(list(self.projector.modules())):
            if isinstance(module, nn.Linear): return int(module.out_features)
        return S2_DIM if isinstance(self.projector, nn.Identity) else -1

    def forward(self, images: Tensor) -> Tensor:
        original = images.shape
        flat = official_preprocess(images).reshape(-1, 3, LEWM_IMAGE_SIZE, LEWM_IMAGE_SIZE)
        with torch.no_grad():
            output = self.encoder(flat, interpolate_pos_encoding=True)
            latent = self.projector(output.last_hidden_state[:, 0])
        return latent.reshape(*original[:-3], S2_DIM)


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


class H3Layer1VelocityField(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__(); self.width = width; self.condition_dim = 3 * LATENT_DIM
        self.input_projection = nn.Linear(LATENT_DIM, width)
        self.time_embedding = FourierTimeEmbedding(64); self.time_projection = nn.Linear(64, width)
        self.condition_projection = nn.Linear(self.condition_dim, width)
        self.blocks = nn.ModuleList(FlowBlock(width) for _ in range(FLOW_LAYERS))
        self.norm = nn.LayerNorm(width); self.output_projection = nn.Linear(width, LATENT_DIM)

    def forward(self, latent: Tensor, tau: Tensor, head: Tensor, agent: Tensor, state_condition: Tensor) -> Tensor:
        value = self.input_projection(latent)
        condition = self.time_projection(self.time_embedding(tau)) + self.condition_projection(torch.cat((head, agent, state_condition), -1))
        for block in self.blocks: value = block(value, condition)
        return self.output_projection(self.norm(value))


class WindowMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(L2_WINDOW * S2_DIM, LATENT_DIM), nn.GELU(), nn.LayerNorm(LATENT_DIM))
    def forward(self, value: Tensor) -> Tensor: return self.net(value.flatten(1))


class ActionHistoryMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(L2_WINDOW * ACTION_DIM + 1, LATENT_DIM), nn.GELU(), nn.LayerNorm(LATENT_DIM))
    def forward(self, actions: Tensor, mask: Tensor) -> Tensor:
        return self.net(torch.cat(((actions * mask[:, None, :]).flatten(1), mask), -1))


class StateVelocityField(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.time = FourierTimeEmbedding(64)
        dims = (S2_DIM + LATENT_DIM + LATENT_DIM + S2_DIM + 1 + 1 + 64, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, S2_DIM)
        layers: list[nn.Module] = []
        for index, (before, after) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(before, after))
            if index < 3: layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
    def forward(self, x: Tensor, tau: Tensor, window: Tensor, actions: Tensor, goal: Tensor, mask: Tensor, remaining: Tensor) -> Tensor:
        return self.net(torch.cat((x, window, actions, goal, mask, remaining, self.time(tau)), -1))


class StateSubgoalGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.window_encoder = WindowMLP(); self.action_encoder = ActionHistoryMLP(); self.velocity = StateVelocityField()

    def condition(self, window: Tensor, actions: Tensor, goal: Tensor, mask: Tensor, remaining: Tensor):
        return self.window_encoder(window), self.action_encoder(actions.detach(), mask), goal, mask, remaining

    def integrate(self, x0: Tensor, condition: tuple[Tensor, ...], steps: int = EULER_STEPS) -> tuple[Tensor, Tensor]:
        x = x0; length = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype); dt = 1.0 / steps
        for index in range(steps):
            tau = torch.full((x.shape[0], 1), index * dt, device=x.device, dtype=x.dtype)
            delta = dt * self.velocity(x, tau, *condition); x = x + delta; length = length + delta.norm(dim=-1)
        return x, length

    def losses(self, x0: Tensor, x1: Tensor, condition: tuple[Tensor, ...], tau: Tensor) -> dict[str, Tensor]:
        mixed = (1 - tau) * x0 + tau * x1
        flow = F.mse_loss(self.velocity(mixed, tau, *condition), x1 - x0)
        endpoint, path = self.integrate(x0, condition)
        endpoint_loss = F.mse_loss(endpoint, x1)
        return {"flow": flow, "endpoint": endpoint_loss, "total": flow + endpoint_loss, "prediction": endpoint, "path_length": path}

class DirectSubgoalGenerator(nn.Module):
    """G2-only endpoint residual control with identical condition encoders/width/depth."""
    def __init__(self) -> None:
        super().__init__(); self.window_encoder=WindowMLP(); self.action_encoder=ActionHistoryMLP(); self.regressor=StateVelocityField()
    def condition(self,window,actions,goal,mask,remaining): return self.window_encoder(window),self.action_encoder(actions.detach(),mask),goal,mask,remaining
    def forward(self,x0,condition):
        tau=torch.zeros((x0.shape[0],1),device=x0.device,dtype=x0.dtype); return x0+self.regressor(x0,tau,*condition)


class H3Policy(nn.Module):
    def __init__(self, config: H3ModelConfig, *, lewm: FrozenLeWMEncoder | None = None) -> None:
        super().__init__(); self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.lewm = lewm if lewm is not None else FrozenLeWMEncoder()
            self.head_encoder = ImageSequenceEncoder(HISTORY_STEPS, LATENT_DIM)
            self.agent_encoder = ConvSequenceEncoder(ACTION_DIM, LATENT_DIM, HIDDEN_DIM)
            self.action_autoencoder = VariationalActionAutoencoder(LATENT_DIM, HIDDEN_DIM)
            self.state_projection = nn.Linear(S2_DIM, LATENT_DIM)
            width = HIDDEN_DIM if config.hierarchical else int(config.flat_hidden_dim)
            self.layer1_velocity = H3Layer1VelocityField(width)
            self.layer2 = StateSubgoalGenerator() if config.hierarchical else None

    def encode_s2(self, images: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        return (self.lewm(images) - mu) / sigma

    def predict_s2(self, batch: Mapping[str, Tensor], mu: Tensor, sigma: Tensor) -> Tensor:
        if self.layer2 is None: raise RuntimeError("flat model has no layer two")
        window = self.encode_s2(batch["observation_window"], mu, sigma)
        x0 = self.encode_s2(batch["current_image"], mu, sigma)
        goal = self.encode_s2(batch["goal_image"], mu, sigma)
        condition = self.layer2.condition(window, batch["action_history"], goal, batch["history_mask"], batch["remaining_horizon"])
        return self.layer2.integrate(x0, condition)[0]

    def layer2_loss(self, batch: Mapping[str, Tensor], mu: Tensor, sigma: Tensor, generator: torch.Generator) -> dict[str, Tensor]:
        assert self.layer2 is not None
        window = self.encode_s2(batch["observation_window"], mu, sigma)
        x0 = self.encode_s2(batch["current_image"], mu, sigma)
        x1 = self.encode_s2(batch["teacher_image"], mu, sigma).detach()
        goal = self.encode_s2(batch["goal_image"], mu, sigma)
        condition = self.layer2.condition(window, batch["action_history"], goal, batch["history_mask"], batch["remaining_horizon"])
        tau = torch.rand((x0.shape[0], 1), device=x0.device, generator=generator)
        return self.layer2.losses(x0, x1, condition, tau)

    def _integrate_l1(self, start: Tensor, head: Tensor, agent: Tensor, state: Tensor) -> Tensor:
        x = start; dt = 1 / EULER_STEPS
        for index in range(EULER_STEPS):
            tau = torch.full((x.shape[0], 1), index * dt, device=x.device, dtype=x.dtype)
            x = x + dt * self.layer1_velocity(x, tau, head, agent, state)
        return x

    def layer1_loss(self, batch: Mapping[str, Tensor], state_std: Tensor, generator: torch.Generator) -> dict[str, Tensor]:
        head = self.head_encoder(batch["head_cam"]); agent = self.agent_encoder(batch["agent_pos"])
        start = agent; target, mean, logvar = self.action_autoencoder.sample(batch["future_actions"], generator)
        tau = torch.rand((start.shape[0], 1), device=start.device, generator=generator)
        state = self.state_projection(state_std)
        velocity = self.layer1_velocity((1-tau)*start + tau*target, tau, head, agent, state)
        flow = F.mse_loss(velocity, target-start); endpoint = self._integrate_l1(start, head, agent, state)
        latent = F.mse_loss(endpoint, target); flow_action = F.l1_loss(self.action_autoencoder.decode(endpoint), batch["future_actions"]); teacher_action = F.l1_loss(self.action_autoencoder.decode(target), batch["future_actions"])
        kl = -.5 * torch.mean(1 + logvar - mean.square() - logvar.exp())
        total = flow + latent + .5*flow_action + .5*teacher_action + KL_WEIGHT*kl
        return {"total": total, "flow": flow, "latent_consistency": latent, "flow_action_l1": flow_action, "teacher_action_l1": teacher_action, "kl": kl}

    @torch.no_grad()
    def predict_actions(self, head_cam: Tensor, agent_pos: Tensor, state_std: Tensor) -> Tensor:
        head = self.head_encoder(head_cam); agent = self.agent_encoder(agent_pos); state = self.state_projection(state_std)
        return self.action_autoencoder.decode(self._integrate_l1(agent, head, agent, state))


def select_flat_width(hierarchical: H3Policy, maximum_width: int = 8192) -> dict[str, int | float]:
    target = int(parameter_manifest(hierarchical)["trainable"])
    current_velocity = sum(p.numel() for p in hierarchical.layer1_velocity.parameters())
    shared = target - current_velocity - sum(p.numel() for p in hierarchical.layer2.parameters())
    candidates = [(abs(shared + layer1_velocity_parameter_count(LATENT_DIM, width, 3*LATENT_DIM) - target), width, shared + layer1_velocity_parameter_count(LATENT_DIM, width, 3*LATENT_DIM)) for width in range(1, maximum_width+1)]
    difference, width, count = min(candidates)
    relative = difference / target
    if relative > .001: raise RuntimeError(f"h3_input_blocked: closest flat capacity differs by {relative:.8f}")
    return {"width": width, "hierarchical_trainable": target, "flat_trainable": count, "absolute_difference": difference, "relative_difference": relative}


def build_capacity_matched_pair(seed: int, lewm: FrozenLeWMEncoder | None = None):
    hierarchical = H3Policy(H3ModelConfig(seed=seed), lewm=lewm)
    selection = select_flat_width(hierarchical)
    flat = H3Policy(H3ModelConfig(seed=seed, hierarchical=False, flat_hidden_dim=int(selection["width"])), lewm=lewm)
    observed = parameter_manifest(flat)["trainable"]
    if observed != selection["flat_trainable"]: raise RuntimeError("h3_input_blocked: flat analytical parameter count mismatch")
    return hierarchical, flat, {"selection": selection, "hierarchical": parameter_manifest(hierarchical), "flat": parameter_manifest(flat)}


def save_checkpoint(path: Path, model: H3Policy, update: int, ema_state: Mapping[str, Tensor]) -> None:
    if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": CHECKPOINT_VERSION, "experiment": EXPERIMENT_ID, "config": asdict(model.config), "update": update, "ema_state": dict(ema_state), "lewm_sha256": module_state_sha256(model.lewm)}, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[H3Policy, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_VERSION or payload.get("experiment") != EXPERIMENT_ID: raise ValueError("unsupported H3 checkpoint")
    model = H3Policy(H3ModelConfig(**payload["config"])); model.load_state_dict(payload["ema_state"])
    return model, {key: payload[key] for key in ("update", "config", "lewm_sha256")}
