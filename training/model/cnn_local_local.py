from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


class CNN_LOCAL_LOCAL(TFModelV2):
    # Fully connected layer
    """Multi-agent model that implements a centralized value function.
    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).
    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CNN_LOCAL_LOCAL, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        """ ACTION MODEL """
        input_image_action = Input(shape=(18, 18, 4), name="input_image")

        conv_1_action = Conv2D(4, (3, 3), padding="same", activation="relu", strides=1)(input_image_action)

        flat_action = Flatten()(conv_1_action)
        dense_layer_action = Dense(128)(flat_action)
        dense_layer_action2 = Dense(128)(dense_layer_action)
        dense_layer_critic = Dense(128)(flat_action)
        dense_layer_critic2 = Dense(128)(dense_layer_critic)

        outputs_action = Dense(num_outputs)(dense_layer_action2)
        outputs_value = Dense(1, name="out_value", activation=None)(dense_layer_critic2)

        self.action_model = tf.keras.Model(input_image_action, [outputs_action, outputs_value])

    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.action_model([input_dict['obs']['d_image']])
        _, self._value_out = self.action_model([input_dict['obs']['e_image_critic']])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
