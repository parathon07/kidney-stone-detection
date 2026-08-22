import tensorflow as tf
from typing import Dict, Any


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
            x = tf.keras.layers.BatchNormalization(name=f"block{block_idx+1}_bn{conv_idx+1}")(x)
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
