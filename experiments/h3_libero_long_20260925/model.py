"""H3 control topology with frozen V-JEPA frame features and T5 goals.

The unchanged H3/H1 modules are bundled under lineage/ with hashes. H3's
lower losses, integration, flow blocks and action VAE are inherited, not
replaced by the failed joint-modality Transformer.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
LINEAGE = HERE / 'lineage'
if LINEAGE.exists():
    sys.path.insert(0, str(LINEAGE))
else:
    sys.path.insert(0, str(HERE.parent / 'a2a_h3_state_subgoal_pusht'))
import h3_model as h3
import h1_model as h1


@dataclass(frozen=True)
class Config:
    hierarchical: bool = True
    seed: int = 42
    flat_width: int | None = None
    visual_dim: int = 2048
    text_dim: int = 768
    state_dim: int = 9
    action_dim: int = 7


class SequenceEncoder(h1.ConvSequenceEncoder):
    def __init__(self, input_dim):
        super().__init__(input_dim, 512, 512)
        self.input_dim = input_dim

    def forward(self, value):
        if value.ndim != 3 or value.shape[1:] != (8, self.input_dim):
            raise ValueError('invalid eight-step sequence')
        return self.projection(self.encoder(value.transpose(1, 2)).flatten(1))


class ActionVAE(h1.VariationalActionAutoencoder):
    def __init__(self, action_dim):
        super().__init__(512, 512)
        self.conv_entry = SequenceEncoder(action_dim)
        self.action_tokens = nn.Linear(action_dim, 512)
        self.action_head = nn.Linear(512, action_dim)


class ActionHistory(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16 * action_dim + 1, 512), nn.GELU(), nn.LayerNorm(512))

    def forward(self, actions, mask):
        return self.net(torch.cat(((actions * mask[:, None, :]).flatten(1), mask), -1))


class StateVelocity(h3.StateVelocityField):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.time = h1.FourierTimeEmbedding(64)
        dims = (config.visual_dim + 512 + 512 + config.text_dim + 2 + 64,
                512, 512, 512, config.visual_dim)
        layers = []
        for i, (before, after) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(before, after))
            if i < 3:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)


class Upper(h3.StateSubgoalGenerator):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.window_encoder = nn.Sequential(nn.Flatten(1), nn.Linear(16 * config.visual_dim, 512),
                                            nn.GELU(), nn.LayerNorm(512))
        self.action_encoder = ActionHistory(config.action_dim)
        self.velocity = StateVelocity(config)


class Policy(h3.H3Policy):
    def __init__(self, config):
        nn.Module.__init__(self)
        if config.hierarchical == (config.flat_width is not None):
            raise ValueError('only flat policy specifies width')
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.head_encoder = nn.Sequential(nn.Flatten(1), nn.Linear(8 * config.visual_dim, 512))
            self.agent_encoder = SequenceEncoder(config.state_dim)
            self.action_autoencoder = ActionVAE(config.action_dim)
            self.state_projection = nn.Linear(config.visual_dim if config.hierarchical else config.text_dim, 512)
            self.layer1_velocity = h3.H3Layer1VelocityField(512 if config.hierarchical else config.flat_width)
            self.layer2 = Upper(config) if config.hierarchical else None

    def upper_condition(self, batch):
        return self.layer2.condition(batch['upper_visual'], batch['upper_actions'], batch['language'],
                                     batch['upper_mask'], batch['remaining'])

    def predict_subgoal(self, batch):
        if self.layer2 is None:
            raise ValueError('flat policy has no upper state')
        return self.layer2.integrate(batch['upper_visual'][:, -1], self.upper_condition(batch))[0]

    def upper_loss(self, batch, generator):
        tau = torch.rand((len(batch['language']), 1), device=batch['language'].device, generator=generator)
        return self.layer2.losses(batch['upper_visual'][:, -1], batch['upper_teacher'].detach(),
                                  self.upper_condition(batch), tau)

    def lower_loss(self, batch, condition, generator):
        return self.layer1_loss(dict(head_cam=batch['lower_visual'], agent_pos=batch['lower_states'],
                                    future_actions=batch['future_actions']), condition, generator)

    @torch.no_grad()
    def actions(self, batch, condition):
        return self.predict_actions(batch['lower_visual'], batch['lower_states'], condition)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_configs(seed=42):
    hier = Policy(Config(seed=seed))
    target = count_parameters(hier)
    flat = Policy(Config(hierarchical=False, seed=seed, flat_width=512))
    shared = count_parameters(flat) - sum(p.numel() for p in flat.layer1_velocity.parameters())
    delta, width, count = min((abs(shared + h1.layer1_velocity_parameter_count(512, w, 1536) - target), w,
                              shared + h1.layer1_velocity_parameter_count(512, w, 1536)) for w in range(1, 4097))
    if delta / target > .001:
        raise ValueError('capacity matching exceeds 0.1 percent')
    return dict(hierarchical=asdict(Config(seed=seed)), flat=asdict(Config(hierarchical=False, seed=seed, flat_width=width))), dict(hierarchical=target, flat=count, relative_difference=delta/target)
