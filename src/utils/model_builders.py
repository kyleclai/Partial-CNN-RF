"""
Model architecture builders for CNN baselines.
Supports VGG16 (pretrained ImageNet) and LeNet (from scratch).
"""

import tensorflow as tf
from keras import layers, models


def build_vgg16(input_shape=(128, 128, 3), num_classes=1, freeze_base=True):
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
    
    # Freeze convolutional layers for transfer learning
    base_model.trainable = not freeze_base
    
    # Build full model with custom classification head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # Always use pretrained weights in inference mode
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
    x = layers.Dropout(0.25)(x)  # ADD THIS
    
    # Conv Block 2
    x = layers.Conv2D(16, (5, 5), activation='relu', name='conv_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool_2')(x)
    x = layers.Dropout(0.25)(x)  # ADD THIS

    # Dense Layers
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(120, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.5)(x)  # ADD THIS
    x = layers.Dense(84, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.5)(x)  # ADD THIS

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


# def get_strategic_vgg16_layers():
#     """
#     Return strategic VGG16 convolutional block endpoints for feature extraction.
#     These correspond to the end of each major conv block before pooling.
    
#     Returns:
#         List of layer names: ['block1_conv2', 'block2_conv2', 'block3_conv3', 
#                                'block4_conv3', 'block5_conv3']
#     """
#     return [
#         # 'block1_conv2',  # 64 filters, large spatial resolution
#         # 'block2_conv2',  # 128 filters
#         'block3_conv3',  # 256 filters
#         'block4_conv3',  # 512 filters
#         'block5_conv3',  # 512 filters, most abstract
#     ]


def get_all_vgg16_conv_layers():
    """
    Return ALL VGG16 convolutional layer names for comprehensive analysis.
    With GAP, we can now extract from all 13 conv layers without OOM.
    
    Returns:
        List of all conv layer names in VGG16
    """
    return [
        'block1_conv1',  # 64 filters
        'block1_conv2',  # 64 filters
        'block2_conv1',  # 128 filters
        'block2_conv2',  # 128 filters
        'block3_conv1',  # 256 filters
        'block3_conv2',  # 256 filters
        'block3_conv3',  # 256 filters
        'block4_conv1',  # 512 filters
        'block4_conv2',  # 512 filters
        'block4_conv3',  # 512 filters
        'block5_conv1',  # 512 filters
        'block5_conv2',  # 512 filters
        'block5_conv3',  # 512 filters
    ]


def create_feature_extractor(base_model, layer_name):
    """
    Create a new model that outputs activations from a specific layer.
    Handles nested VGG16 models correctly by rebuilding the extraction path.
    
    Args:
        base_model: Trained Keras model
        layer_name: Name of target layer
    
    Returns:
        Keras Model that outputs layer activations
    """
    try:
        # Direct access for flat models (LeNet)
        layer_output = base_model.get_layer(layer_name).output
        extractor = models.Model(
            inputs=base_model.input,
            outputs=layer_output
        )
        return extractor
    except ValueError:
        # Nested VGG16 case: extract from the VGG16 submodel directly
        try:
            vgg16_submodel = base_model.get_layer('vgg16')
            
            # Build a new standalone extractor using only the VGG16 submodel
            # This avoids the graph-tracing issue
            layer_output = vgg16_submodel.get_layer(layer_name).output
            
            extractor = models.Model(
                inputs=vgg16_submodel.input,
                outputs=layer_output
            )
            return extractor
            
        except (ValueError, AttributeError) as e:
            available_layers = get_all_layer_names(base_model)
            
            try:
                vgg16_model = base_model.get_layer('vgg16')
                vgg16_layers = [layer.name for layer in vgg16_model.layers]
                raise ValueError(
                    f"Layer '{layer_name}' not found. "
                    f"Top-level: {available_layers}. "
                    f"VGG16 internal: {vgg16_layers[:10]}..."
                ) from e
            except:
                raise ValueError(
                    f"Layer '{layer_name}' not found. Available: {available_layers}"
                ) from e