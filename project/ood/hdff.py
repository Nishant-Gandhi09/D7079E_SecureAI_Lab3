import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
 
from functools import partial
from tqdm import tqdm
 
from functools import partial
 
from ood.VSA import Vsa
from config import ConfigDataset, ConfigOod
 
# ----------------------------------------------------------------------------
# Implementation Checklist
# ----------------------------------------------------------------------------
# - [x] Dummy input shape matches (batch_size, *input_shape)
# - [x] self.features length equals number of model layers
# - [x] Projection matrices match each layer's channel count
# - [x] All projected features have shape (batch, hyper_size)
# - [x] Bundled vector has shape (batch, hyper_size)
# - [x] Similarity returns a scalar or vector consistently
 
class Hdff():
    # Conceptual role:
    # HDFF (Hyperdimensional Feature Fusion) builds a single high-dimensional
    # signature of a neural network by:
    #   1) extracting intermediate layer outputs
    #   2) projecting each layer's output into a shared hypervector space
    #   3) bundling (superposing) these projected vectors
    # The result is a compact representation that can be compared across models
    # (e.g., global vs. local in federated learning) using a similarity function.
    #
    # Key theory ideas:
    # - Hyperdimensional computing uses very high dimensional vectors (e.g., 1k-10k)
    #   that can be combined and compared robustly using simple operations like
    #   bundling (addition/superposition) and similarity (cosine or dot-product).
    # - Neural networks produce features at multiple levels of abstraction. HDFF
    #   fuses them into a single vector that still reflects the multi-layer structure.
    # - A fixed projection per layer creates a stable mapping so different models
    #   can be compared in the same hypervector space.
    def __init__(self, ood_config : ConfigOod, dataset_config : ConfigDataset):
        # - Store config objects for later use (e.g., hypervector size, debug flags).
        # - Initialize lists for projections, feature tensors, and results.
        # - Create a dummy input tensor with shape:
        #       (batch_size, *input_shape)
        #   Use ones or random values; only shape matters.
        # - Instantiate the VSA helper (may take a debug flag).
        """ Hyperdimensional feature fusion.
            Inspect the paper https://arxiv.org/abs/2112.05341 to understand the math behind.
            We apply the feature fusion for OOD detection between two models in federated learning.
            Its a bit different from the paper as we only compare the models output feature vectors.
            But the fundamental theory are the same.
            No data is needed for this approach, only the models.
 
        Args:
            ood_config (ConfigHdff): Hyperdimensional configuration.
            dataset_config (ConfigDataset): Dataset configuration.
        """
        self.vsa = Vsa(debug=ood_config.hdc_debug)
        self.dataset_config = dataset_config
        self.ood_config = ood_config
 
        self.proj = []
        self.features = [0]
        self.results = []
 
        input_shape = (self.dataset_config.batch_size,) + self.dataset_config.input_shape
        self.dummy_input = tf.ones(input_shape)
 
    def feature_update(self, model : tf.keras.models.Sequential):
        # Theory:
        # This step actually computes the layer outputs and stores them. It runs the
        # model on a dummy input and collects the output of every layer.
        #
        # Why it exists:
        # - Projection and bundling require actual tensors from the model.
        # - Using a dummy input ensures shape-correct outputs even if you do not have
        #   real data available in the lab.
        #
        # Implementation hints:
        # - Build a model that outputs all layers at once. In Keras, this is typically
        #   done by specifying outputs=[layer.output for layer in model.layers].
        # - Run the dummy input through this multi-output model.
        # - Store each result into self.features[i].
        # - Make sure the ordering matches the earlier layer list.
        #
        """ Update feature vector with dummy input from dataset config with feature output vector from model.
            You can use a dummy input through the model to get the output feature vectors.
            Use tf.ones(input_shape).
 
        Args:
            model (tf.keras.models.Sequential): Model for extracting output feature vectors.
        """
        multi_output_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[layer.output for layer in model.layers]
        )
        outputs = multi_output_model(self.dummy_input, training=False)
        for i, out in enumerate(outputs):
            self.features[i] = out
 
 
    def feature_extraction(self, model : tf.keras.models.Sequential):
        # Theory:
        # This step inspects the model structure and prepares storage for layer outputs.
        # It does NOT run data through the model. It is a preparation stage.
        #
        # Why it exists:
        # - You need to know how many layers you will capture and how many feature
        #   tensors to expect.
        # - It provides a consistent index mapping between model layers and stored
        #   feature tensors.
        #
        # Implementation hints:
        # - Iterate through model.layers, store their references or names.
        # - Count layers and allocate self.features as a list of that length.
        # - Optional: if debug is enabled, print layer names and indices.
        # - Avoid extracting features here; this should only configure structure.
        """ Count layers and create feature vector based on models layers and structure.
 
        Args:
            model (tf.keras.models.Sequential): _description_
        """
        self.features = [None] * len(model.layers)
        if self.ood_config.hdc_debug:
            for i, layer in enumerate(model.layers):
                print(f"[HDFF] Layer {i}: {layer.name} — output shape: {layer.output_shape}")
 
    def feature_bundle(self, debug : bool):
        # Theory:
        # Bundling is the fusion stage. Each layer's feature is compressed, projected
        # into the hypervector space, and then combined into a single representation.
        #
        # Why it exists:
        # - You want a compact, fixed-size signature of the entire model.
        # - Bundling preserves information across layers without exploding dimension.
        #
        # Implementation hints:
        # - For each feature tensor:
        #     1) If it has spatial dimensions (e.g., 4D), reduce it to (batch, channels)
        #        using average pooling or global average pooling.
        #     2) Project it with its layer-specific matrix:
        #            projected = feature @ proj_matrix
        #     3) Combine with the running bundle. If it is the first layer, initialize
        #        the bundle with its projected vector; otherwise use VSA.bundle.
        # - Debug mode can print or log the projected vectors.
        # - Ensure all projected vectors have shape (batch, hyper_size).
        # - Decide whether bundling should be a simple sum or a VSA method (depends on
        #   your VSA helper class). If summing, you may normalize afterward.
        #
        """ Project output feature vector onto projection matrix, into high dimensional space.
            Creating feature bundle for each layer.
 
        Returns:
            tensor: bundled hypervector of shape (batch, hyper_size)
        """
        bundle = None
        for i, (feature, proj_matrix) in enumerate(zip(self.features, self.proj)):
            feature = tf.cast(feature, tf.float32)
 
            # Global average pool spatial dimensions for 4D tensors (batch, H, W, C) → (batch, C)
            if len(feature.shape) == 4:
                feature = tf.reduce_mean(feature, axis=[1, 2])
 
            # Project: (batch, channels) @ (channels, hyper_size) → (batch, hyper_size)
            projected = tf.linalg.matmul(feature, tf.cast(proj_matrix, tf.float32))
 
            if debug:
                print(f"[HDFF] Layer {i}: projected shape = {projected.shape}")
 
            if bundle is None:
                bundle = projected
            else:
                bundle = self.vsa.bundle(bundle, projected)  # superposition (addition)
 
        return bundle
 
    def projection_matrices(self):
        # Theory:
        # Each layer output has its own dimensionality. To combine them, you project
        # each layer into the same hypervector space of size hyper_size.
        #
        # Why it exists:
        # - You cannot bundle vectors of different sizes directly.
        # - A fixed projection per layer makes comparisons between different models
        #   meaningful and repeatable.
        #
        # Implementation hints:
        # - For each feature tensor, determine the channel dimension:
        #     - If feature is 4D (e.g., NHWC), channels = shape[3].
        #     - If feature is 2D (batch, channels), channels = shape[1].
        # - Create a projection matrix of shape (channels, hyper_size).
        # - Use a stable initializer like orthogonal or random normal.
        # - Store the list of matrices in self.proj.
        # - These matrices should be fixed (non-trainable) for consistent similarity.
        #
        """ Create projection matrix.
        """
        self.proj = []
        initializer = tf.initializers.GlorotUniform()
        for feature in self.features:
            # Determine channel dimension
            if len(feature.shape) == 4:   # (batch, H, W, C)
                channels = feature.shape[3]
            else:                          # (batch, units)
                channels = feature.shape[-1]
 
            proj_matrix = tf.Variable(
                initializer(shape=(channels, self.ood_config.hyper_size), dtype=tf.float32),
                trainable=False,
                dtype=tf.float32
            )
            self.proj.append(proj_matrix)
 
    def similarity(self, bundle1, bundle2):
        # Theory:
        # This computes how close two bundled hypervectors are. Similarity is often
        # cosine similarity or dot product after normalization. High similarity means
        # the two models share similar feature signatures.
        #
        # Why it exists:
        # - It provides a simple metric to compare client vs. global models or to
        #   detect changes in a model over time.
        #
        # Implementation hints:
        # - Use a VSA helper method like vsa.similarity for cosine similarity.
        # - Return a single scalar (e.g., tf.math.reduce_max similarity in batch) or a vector of
        #   similarity values. Document your choice clearly.
        # - Debug mode can print the input bundles and summary statistics.
        """ Cosine similarity on two feature bundles.
            Should be between global model and local model x, bundle 1 and bundle 2.
            Returns the max cosine similarity across the batch dimension as a scalar.
 
        Args:
            bundle1 (tensor): Projection matrix 1 with output feature projection.
            bundle2 (tensor): Projection matrix 2 with output feature projection.
 
        Returns:
            tensor: Scalar cosine similarity value.
        """
        if self.ood_config.hdc_debug:
            print(f"[HDFF] Similarity — bundle1: {bundle1.shape}, bundle2: {bundle2.shape}")
 
        sim_matrix = self.vsa.similarity(bundle1, bundle2)  # (batch1, batch2)
        return tf.math.reduce_max(sim_matrix)
 
    def set_projection_matrices(self, proj):
        # Theory:
        # This function allows you to inject externally created projection matrices.
        # This is useful in federated settings where you want all clients to share
        # the same projection basis.
        #
        # Implementation hints:
        # - Validate the provided projection list has the same length as features.
        # - Simply assign it to self.proj.
        # - Optional: add shape checks for safety (channels x hyper_size).
        """Set projection matrix to arg.
 
        Args:
            proj (tensor): Projection matrix to update.
        """
        assert len(proj) == len(self.features), \
            f"[HDFF] Projection matrices length {len(proj)} must match features length {len(self.features)}"
        self.proj = proj
 
    def set_dummy_input(self, dummy_input):
        # Theory:
        # This allows the caller to replace the default dummy input with a custom one.
        # This is helpful if batch size or input shape changes later.
        #
        # Implementation hints:
        # - Simply assign the provided tensor to self.dummy_input.
        # - Optional: validate the shape matches expected model input.
        """Set dummy input to arg.
 
        Args:
            dummy_input (tensor): Dummy input to update.
        """
        self.dummy_input = dummy_input