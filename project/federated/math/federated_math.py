import tensorflow as tf

def federated_mean(local_weights: list):
    """
    FedAvg averaging for a single layer.

    Takes the weight matrix for ONE layer collected from every participating
    local model, stacks them into a single tensor, and returns the element-wise
    mean across all models (axis=0).

    Args:
        local_weights (list): List of weight arrays — one entry per local model,
                              all for the same layer.

    Returns:
        tf.Tensor: Averaged weight tensor with the same shape as each input array.
    """
    # Cast to float32 for numerical consistency, then stack along a new axis 0
    # so shape becomes (num_models, *layer_shape).
    stacked = tf.stack([tf.cast(w, tf.float32) for w in local_weights], axis=0)

    # Reduce mean across the model axis → result shape is (*layer_shape,).
    return tf.math.reduce_mean(stacked, axis=0)