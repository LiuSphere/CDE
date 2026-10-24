"""H10: recode only the generator-fraction input to the local Critic."""

import numpy as np
import torch
from gymnasium.spaces import Box

from modules.critic import GAVDACCritic


class CenteredGeneratorGAVDACCritic(GAVDACCritic):
    def __init__(self, scheme, args):
        if args.n_agents != 3 or args.n_actions != 2:
            raise ValueError("H10 requires three agents and two fraction actions")
        spaces = args.action_spaces
        if len(spaces) != 3 or any(
            not isinstance(space, Box) or space.shape != (2,)
            or not np.array_equal(space.low, [0.0, -1.0])
            or not np.array_equal(space.high, [1.0, 1.0])
            for space in spaces
        ):
            raise ValueError("H10 requires identical fraction Boxes [0,-1]/[1,1]")
        super().__init__(scheme, args)
        self.centered_forward_calls = 0

    def forward(self, inputs, actions, hidden_state=None):
        if actions is None:
            raise ValueError("H10 requires separate observation and action tensors")
        if actions.ndim < 2 or actions.shape[-2:] != (3, 2):
            raise ValueError("H10 action shape must end in agents=3, actions=2")
        if (inputs.ndim != 2 or inputs.shape[1] != self.input_shape - 2
                or inputs.shape[0] != actions.numel() // 2):
            raise ValueError("H10 requires flattened observations in matching B/T/N order")
        if any(value.device.type != "cpu" or value.dtype != torch.float32
               for value in (inputs, actions)):
            raise ValueError("H10 observations/actions must be CPU float32")
        if not torch.isfinite(actions).all():
            raise ValueError("H10 actions must be finite")
        centered = torch.stack((2 * actions[..., 0] - 1, actions[..., 1]), dim=-1)
        output = super().forward(inputs, centered, hidden_state)
        self.centered_forward_calls += 1
        return output
