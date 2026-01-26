"""
Model architecture builders for CNN baselines.
Supports VGG16 (pretrained ImageNet) and LeNet (from scratch).
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_vgg16(input_shape=(128, 128, 3), num_classes=1, freeze_base=False):
    """
    Build VGG16 with ImageNet pretrained weights.
    
    Args:
        input_shape: Input image dimensions (H, W, C)
        num_classes: Number of output classes (1 for binary)
        freeze_base: If True, freeze convolutional base for transfer learning
    
    Returns:
        Compiled Keras Model
    """
    # Load VGG16 pretrained on ImageNet, exclude top layers
    base_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Optionally freeze convolutional layers
    if freeze_base:
        base_model.trainable = False
    
    # Build full model with custom classification head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=not freeze_base)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.5, name='dropout2')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def build_lenet(input_shape=(128, 128, 3), num_classes=1):
    """
    Build LeNet-style CNN from scratch (matches original notebook architecture).
    
    Args:
        input_shape: Input image dimensions (H, W, C)
        num_classes: Number of output classes (1 for binary)
    
    Returns:
        Keras Model (uncompiled)
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv2D(6, (5, 5), activation='relu', name='conv_1')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool_1')(x)
    
    # Conv Block 2
    x = layers.Conv2D(16, (5, 5), activation='relu', name='conv_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool_2')(x)
    
    # Dense Layers
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(120, activation='relu', name='dense_1')(x)
    x = layers.Dense(84, activation='relu', name='dense_2')(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def get_all_layer_names(model):
    """
    Extract all layer names from a Keras model (excluding Input layers).
    
    Args:
        model: Keras Model
    
    Returns:
        List of layer names suitable for feature extraction
    """
    return [
        layer.name for layer in model.layers 
        if 'input' not in layer.name.lower()
    ]


def create_feature_extractor(base_model, layer_name):
    """
    Create a new model that outputs activations from a specific layer.
    
    Args:
        base_model: Trained Keras model
        layer_name: Name of target layer
    
    Returns:
        Keras Model that outputs layer activations
    """
    try:
        layer_output = base_model.get_layer(layer_name).output
        extractor = models.Model(
            inputs=base_model.input,
            outputs=layer_output
        )
        return extractor
    except ValueError as e:
        raise ValueError(
            f"Layer '{layer_name}' not found in model. "
            f"Available layers: {get_all_layer_names(base_model)}"
        ) from e