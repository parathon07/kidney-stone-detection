import tensorflow as tf
import keras
from keras.saving import register_keras_serializable
from typing import Dict, Any


@register_keras_serializable(package="kidney_cnn")
class Fp16BatchNorm(keras.layers.Layer):
    """Batch normalization that keeps activations in the active dtype.

    The layer uses the same channel-wise parameters, epsilon, and momentum as
    stock BatchNormalization, but avoids its full-activation fp32 upcast under
    mixed_float16. Statistics and parameters are kept in fp32 storage and
    explicitly cast for the activation calculation. The non-autocast storage
    is important: Keras otherwise routes assignments to moving statistics
    through an autocast view, leaving them at their initial values.
    """

    def __init__(self, momentum=0.99, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma = self.add_weight(
            name="gamma", shape=(channels,),
            initializer="ones", trainable=True, autocast=False,
        )
        self.beta = self.add_weight(
            name="beta", shape=(channels,),
            initializer="zeros", trainable=True, autocast=False,
        )
        self.moving_mean = self.add_weight(
            name="moving_mean", shape=(channels,),
            initializer="zeros", trainable=False, autocast=False,
        )
        self.moving_variance = self.add_weight(
            name="moving_variance", shape=(channels,),
            initializer="ones", trainable=False, autocast=False,
        )

    def call(self, inputs, training=None):
        # NHWC-native: statistics over (N, H, W) per channel, zero layout
        # changes. Avoids permute copies that autograd must save for backward
        # (~2x activation memory on 384x384 inputs -> VRAM spill on 8 GB GPUs).
        x = inputs
        if training:
            mean = keras.ops.mean(x, axis=[0, 1, 2])
            var = keras.ops.var(x, axis=[0, 1, 2])
            # Moving statistics stored in the variable's dtype (fp32 storage)
            self.moving_mean.assign(
                self.moving_mean * self.momentum
                + keras.ops.cast(mean, self.moving_mean.dtype) * (1.0 - self.momentum)
            )
            self.moving_variance.assign(
                self.moving_variance * self.momentum
                + keras.ops.cast(var, self.moving_variance.dtype) * (1.0 - self.momentum)
            )
        else:
            mean = self.moving_mean
            var = self.moving_variance

        mean_b = keras.ops.reshape(keras.ops.cast(mean, x.dtype), (1, 1, 1, -1))
        var_b = keras.ops.reshape(keras.ops.cast(var, x.dtype), (1, 1, 1, -1))
        gamma_b = keras.ops.reshape(keras.ops.cast(self.gamma, x.dtype), (1, 1, 1, -1))
        beta_b = keras.ops.reshape(keras.ops.cast(self.beta, x.dtype), (1, 1, 1, -1))

        return (x - mean_b) * keras.ops.rsqrt(var_b + self.epsilon) * gamma_b + beta_b

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"momentum": self.momentum, "epsilon": self.epsilon})
        return config


def build_model(config: Dict[str, Any]) -> tf.keras.Model:
    """
    Build the canonical kidney_cnn_v1 model architecture per specification Section 6.

    Structure:
      - Input: 384x384x1 grayscale
      - 4 blocks (filters: 64, 128, 256, 512), 2 convs per block (3x3, stride 1, same padding, use_bias=False, he_normal)
      - Each Conv2D followed by BatchNormalization and ReLU
      - Each block ends with 2x2 MaxPool2D
      - Head: GlobalAveragePooling2D -> Dense(192, he_normal) -> ReLU -> Dropout(0.35) -> Dense(2, Softmax, float32)

    Total expected params: ~4,788,866 (~4,785,026 trainable).
    BatchNormalization is Fp16BatchNorm: identical math/params, fp16-safe execution.
    """
    image_size = config.get("data", {}).get("image_size", 384)
    channels = config.get("data", {}).get("channels", 1)
    model_cfg = config.get("model", {})

    filters_list = model_cfg.get("filters", [64, 128, 256, 512])
    convs_per_block = model_cfg.get("convs_per_block", 2)
    kernel_size = model_cfg.get("kernel_size", 3)
    dense_units = model_cfg.get("dense_units", 192)
    dropout_rate = model_cfg.get("dropout", 0.35)
    model_name = model_cfg.get("name", "kidney_cnn_v1")

    inputs = tf.keras.Input(shape=(image_size, image_size, channels), name="input_image")
    x = inputs

    # 4 Convolutional Blocks
    for block_idx, num_filters in enumerate(filters_list):
        for conv_idx in range(convs_per_block):
            x = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                name=f"block{block_idx+1}_conv{conv_idx+1}",
            )(x)
            x = Fp16BatchNorm(name=f"block{block_idx+1}_bn{conv_idx+1}")(x)
            x = tf.keras.layers.ReLU(name=f"block{block_idx+1}_relu{conv_idx+1}")(x)

        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=2,
            padding="valid",
            name=f"block{block_idx+1}_pool",
        )(x)

    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(
        dense_units,
        use_bias=True,
        kernel_initializer="he_normal",
        name="dense_features",
    )(x)
    x = tf.keras.layers.ReLU(name="dense_relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)

    # Dense(2) Softmax output forced to float32 for mixed precision stability
    outputs = tf.keras.layers.Dense(
        2,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model
