"""
Global pooling utilities for feature extraction.
"""

import tensorflow as tf
from keras import layers, models


def apply_global_pooling(features, pooling_type='avg'):
    """
    Apply global pooling to feature maps.
    
    Args:
        features: Feature tensor, shape (N, H, W, C) or (N, features)
        pooling_type: 'avg' or 'max'
    
    Returns:
        Pooled features, shape (N, C)
    """
    if features.ndim == 2:
        # Already flat (from dense layers)
        return features
    
    if features.ndim == 4:
        # Apply global pooling
        if pooling_type == 'avg':
            # Average across spatial dimensions (H, W)
            return tf.reduce_mean(features, axis=[1, 2])
        elif pooling_type == 'max':
            # Max across spatial dimensions (H, W)
            return tf.reduce_max(features, axis=[1, 2])
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
    else:
        raise ValueError(f"Unexpected feature dimensionality: {features.ndim}D")


def create_feature_extractor_with_pooling(base_model, layer_name, pooling_type='avg'):
    """
    Create feature extractor that applies global pooling after extraction.
    
    Args:
        base_model: Trained Keras model
        layer_name: Name of layer to extract from
        pooling_type: 'avg' or 'max'
    
    Returns:
        Keras Model that outputs pooled features
    """
    from utils.model_builders import create_feature_extractor
    
    # Get base extractor (without pooling)
    base_extractor = create_feature_extractor(base_model, layer_name)
    
    # Add global pooling layer
    inputs = base_extractor.input
    features = base_extractor.output
    
    # Apply pooling if features are 4D (conv outputs)
    if len(features.shape) == 4:
        if pooling_type == 'avg':
            pooled = layers.GlobalAveragePooling2D()(features)
        elif pooling_type == 'max':
            pooled = layers.GlobalMaxPooling2D()(features)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
    else:
        # Already flat (dense layer output)
        pooled = features
    
    return models.Model(inputs=inputs, outputs=pooled)