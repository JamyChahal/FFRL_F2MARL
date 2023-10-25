import json
import math
import random

import numpy as np
import tree

from ray.rllib import SampleBatch

from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights

from behaviors.randomBehavior import RandomBehavior


def make_randomBehavior(map_size):

    class RandomWalkingPolicy(Policy):
        """Hand-coded policy that returns random walking for training."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.behavior = RandomBehavior(map_size=map_size, is_in_training=True)

        @override(Policy)
        def compute_actions(self,
                            obs_batch,
                            state_batches=None,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            **kwargs):
            self.behavior.set_observation(obs_batch[0])
            # Action to perform here
            unbatched = [self.behavior.get_action()]

            actions = tuple(
                np.array([unbatched[j][i] for j in range(len(unbatched))]) for i in range(len(unbatched[0]))
            )

            #actions = unbatched  # Because not a tuple
            return actions, [], {}

        @override(Policy)
        def learn_on_batch(self, samples):
            """No learning."""
            return {}

        @override(Policy)
        def compute_log_likelihoods(self,
                                    actions,
                                    obs_batch,
                                    state_batches=None,
                                    prev_action_batch=None,
                                    prev_reward_batch=None):
            return np.array([random.random()] * len(obs_batch))

        @override(Policy)
        def get_weights(self) -> ModelWeights:
            """No weights to save."""
            return {}

        @override(Policy)
        def set_weights(self, weights: ModelWeights) -> None:
            """No weights to set."""
            pass

        # NEW FUNCTIONS FROM :  https://github.com/ray-project/ray/blob/master/rllib/examples/policy/random_policy.py
        @override(Policy)
        def init_view_requirements(self):
            super().init_view_requirements()
            # Disable for_training and action attributes for SampleBatch.INFOS column
            # since it can not be properly batched.
            vr = self.view_requirements[SampleBatch.INFOS]
            vr.used_for_training = False
            vr.used_for_compute_actions = False

        @override(Policy)
        def compute_log_likelihoods(
                self,
                actions,
                obs_batch,
                state_batches=None,
                prev_action_batch=None,
                prev_reward_batch=None,
        ):
            return np.array([random.random()] * len(obs_batch))

        @override(Policy)
        def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
            return SampleBatch(
                {
                    SampleBatch.OBS: tree.map_structure(
                        lambda s: s[None], self.observation_space.sample()
                    ),
                }
            )

    return RandomWalkingPolicy
