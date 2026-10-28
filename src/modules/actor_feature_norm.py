"""H13: normalize only the learned Actor input features before ReLU."""

import torch
from torch import nn
from torch.nn import functional as F

from modules.policy import RNNAgent


def actor_feature_norm_enabled(args):
    value = getattr(args, "actor_feature_layer_norm", False)
    if type(value) is not bool:
        raise ValueError("actor_feature_layer_norm must be boolean")
    if value:
        required = {"scaled_huber_critic": True, "actor_update_interval": 2,
                    "target_update_interval": 1, "rnn_hidden_dim": 48}
        if any(type(getattr(args, k, None)) is not type(v) or getattr(args, k) != v
               for k, v in required.items()):
            raise ValueError("H13 requires the H12 Actor2/target1/Huber parent and width48")
        if getattr(args, "use_cuda", False):
            raise ValueError("H13 requires CPU float32")
    return value


class LayerNormRNNAgent(RNNAgent):
    def __init__(self, input_shape, args):
        if not actor_feature_norm_enabled(args):
            raise ValueError("H13 Actor requires actor_feature_layer_norm:true")
        super().__init__(input_shape, args)
        self.lnorm = nn.LayerNorm(args.rnn_hidden_dim, eps=1e-5)

    def forward(self, inputs, hidden_state, actions=None):
        x = F.relu(self.lnorm(self.fc1(inputs)))
        hidden = self.rnn(x, hidden_state.reshape(-1, self.args.rnn_hidden_dim))
        return {"actions": torch.tanh(self.fc2(hidden)), "hidden_state": hidden}

    def forward_sequence(self, inputs, hidden_state):
        x = F.relu(self.lnorm(self.fc1(inputs)))
        hidden = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hidden_states = []
        for step in x.unbind(0):
            hidden = self.rnn(step, hidden)
            hidden_states.append(hidden)
        hidden_states = torch.stack(hidden_states, dim=0)
        return {
            "actions": torch.tanh(self.fc2(hidden_states)),
            "hidden_states": hidden_states,
            "hidden_state": hidden,
        }
