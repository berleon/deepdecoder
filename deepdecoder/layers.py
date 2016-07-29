
import keras.backend as K
from keras.layers.core import Layer


class ThresholdBits(Layer):
    def call(self, x, mask=None):
        return 2*K.cast(x > 0, K.floatx()) - 1


class NormSinCosAngle(Layer):
    # TODO: Fix this strange index. Maybe add a ApplyPartially Layer
    def __init__(self, idx, **kwargs):
        self.sin_idx = idx
        self.cos_idx = idx + 1
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        sin = x[:, self.sin_idx:self.sin_idx+1]
        cos = x[:, self.cos_idx:self.cos_idx+1]
        eps = 1e-7
        scale = K.sqrt(1./(eps + sin**2 + cos**2))
        sin_scaled = K.clip(scale*sin, -1, 1)
        cos_scaled = K.clip(scale*cos, -1, 1)
        return K.concatenate([x[:, :self.sin_idx], sin_scaled, cos_scaled,
                              x[:, self.cos_idx+1:]], axis=1)

    def get_config(self):
        config = {'idx': self.sin_idx}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
