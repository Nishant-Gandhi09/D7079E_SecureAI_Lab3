import os
import tensorflow as tf
import copy
import random
import matplotlib.pyplot as plt
 
from config import ConfigDataset, ConfigFederated, ConfigOod, ConfigPlot
from dataset.dataset import Dataset
from federated.math import federated_math
from federated.math.plot import FederatedPlot
 
from ood.hdff import Hdff
from model.model import Model
 
class Federated():
    """
        Federated learning environment. Three cycles per round, update local models from global model, train local models, regression on local models and update global.
    """
    def __init__(self, dataset : Dataset, model : Model, federated_config : ConfigFederated, ood_config : ConfigOod, dataset_config : ConfigDataset, plot_config : ConfigPlot) -> None:
        """
        Args:
            dataset (Dataset): dataset, custom.
            model (Model): nn. model.
            federated_config (ConfigFederated): configuration for federated learning env.
            ood_config (ConfigFederated): configuration for ood detection.
            dataset_config (ConfigDataset) : configuration for dataset.
            plot_config (ConfigPlot): configuration for plotting.
        """
        self.dataset = dataset
        self.init_model = model            # This model can be
        self.federated_config = federated_config
        self.ood_config = ood_config
        self.plot_config = plot_config
        self.models = {}      # model_id: Model instance
        self.datasets = {}    # model_id: dataset iterator(s)
        self.ood_similarity_results = {}  # model_id: list of per-round similarity scores
        self.initialize_models()
 
    def run(self):
        """
            Running federated learning environment.
        """
        # Load pre-trained models if configured
        if self.federated_config.load:
            self._load_models()
            if self.federated_config.load_reg:
                self.regression(self.federated_config.host_id)
 
        for round_idx in range(self.federated_config.rounds):
            print(f"\n--- Round {round_idx + 1}/{self.federated_config.rounds} ---")
 
            # 1. Regression (distribute global weights to all local models)
            self.regression(self.federated_config.host_id)
 
            # 2. Randomly select local clients for this round
            local_clients = [
                i for i in range(self.federated_config.clients)
                if i != self.federated_config.host_id
            ]
            num_participants = min(self.federated_config.participants, len(local_clients))
            selected_clients = random.sample(local_clients, num_participants)
            print(f"[Run] Selected clients for round {round_idx + 1}: {selected_clients}")
 
            # 3. Train selected local models
            self.train(selected_clients, round_idx)
 
            # 4. Aggregation (FedAvg, with optional OOD filtering)
            self.aggregation(selected_clients, round_idx)
 
            # 5. Evaluate global model on its test data
            self.models[self.federated_config.host_id].test(
                self.datasets[self.federated_config.host_id][2]
            )
 
            # 6. Save model weights after each round (if configured)
            if self.federated_config.save:
                self._save_models(round_idx)
 
        # 7. Plot and save results after all rounds
        self.result()
 
    def train_(self, start : int):
        # For training, this will be the functionality flow. However you need to implement them.
 
        for round in range(1+start, self.federated_config.rounds+1):
            part = max(int(self.federated_config.participants), 1)                 # Alteast one client will partcipate in round.
            selected_clients = random.sample(None, part)                           # TODO Select random clients that will participate during training round.
 
            while self.federated_config.host_id in selected_clients:               # If global model gets selected as participant, select new.
                selected_clients = random.sample(None, part)
 
            # This is for ood, if ood is enabled and round is less than ood_round, remove clients that are in ood_client list.
            # Can be good to let the models train for 2 rounds before ood, to get a better model.
            # Some warmup of model before ood and ood client included.
            if(self.ood_config.enabled and round < self.federated_config.ood_round):
                for i in self.ood_config.ood_client:
                    selected_clients.remove(i)
 
            self.global_(self.federated_config.host_id, round)                     # Update all local models with global model.
 
            for id in selected_clients:                                            # Train all local models.
                self.local_(id, round)
 
            self.update_(selected_clients, round)
 
        return round
 
    def test_(self):
        # TODO
        return None
 
    def global_(self, id : int, round : int):                                            # Update all local models with global model weights.
        """
            Updates local models with global model weights.
        Args:
            id (int): id for global model.
        """
        # TODO
        return None
 
    def local_(self, id : int, round : int):                                         # Train local models
        """
            Trains local models, with id.
        Args:
            id (int): id for local model.
            round (int): current round.
        """
        # TODO
        return None
 
    def update_(self, selected_clients, round : int):
        """
            Update global model with clients that was training during round (selected clients).
 
            Incorporate ood detection if enabled in config. Select only clients that are not detected as ood.
        Args:
            selected_clients (list): list with id of clients (local models) that selected for training.
            round (int): current round.
        """
        # TODO
 
        return None
 
 
    def ood_extraction(self, id : int, model : Model):
        """ Extract features from model using HDFF and return the Hdff object
            (features populated, ready for projection_matrices or set_projection_matrices).
 
        Args:
            id (int): Id of model.
            model (Model): Model (object).
 
        Returns:
            Hdff: HDFF object with features extracted from the model.
        """
        hdff = Hdff(ood_config=self.ood_config, dataset_config=self.dataset_config)
        hdff.feature_extraction(model.model)
        hdff.feature_update(model.model)
        return hdff
 
    def ood_detection(self, selected_clients):
        """ Detect OOD clients by comparing each local model's feature bundle
            to the global model's feature bundle via cosine similarity.
 
            The global model's projection matrices are shared with every local
            model so all bundles live in the same hypervector space.
 
        Args:
            selected_clients (list): clients that undergo training this round.
 
        Returns:
            tuple: (filtered_clients, similarities)
                filtered_clients (list): clients that passed the OOD threshold.
                similarities (dict): {client_id: cosine_similarity_score}
        """
        global_id = self.federated_config.host_id
 
        # Step 1: Extract global model's features and build its projection matrices + bundle
        global_hdff = self.ood_extraction(global_id, self.models[global_id])
        global_hdff.projection_matrices()
        global_bundle = global_hdff.feature_bundle(debug=self.ood_config.hdc_debug)
 
        filtered_clients = []
        similarities = {}
 
        for client_id in selected_clients:
            # Step 2: Extract local model's features using the same projection matrices
            local_hdff = self.ood_extraction(client_id, self.models[client_id])
            local_hdff.set_projection_matrices(global_hdff.proj)
            local_bundle = local_hdff.feature_bundle(debug=self.ood_config.hdc_debug)
 
            # Step 3: Cosine similarity — scalar in [0, 1]
            sim = float(global_hdff.similarity(global_bundle, local_bundle).numpy())
            similarities[client_id] = sim
 
            print(f"[OOD] Client {client_id}: cosine similarity = {sim:.4f} "
                  f"(threshold = {self.ood_config.ood_protection_thres})")
 
            # Step 4: Apply security protocol — exclude below threshold if protection enabled
            if not self.ood_config.ood_protection or sim >= self.ood_config.ood_protection_thres:
                filtered_clients.append(client_id)
            else:
                print(f"[OOD] Client {client_id} EXCLUDED from aggregation (OOD detected)")
 
        return filtered_clients, similarities
 
    def result(self):
        """
        Plot performance of each model after all rounds complete, then save
        every figure as a PNG into a 'plots/' directory for the report.
 
        Figure 10 — Global model: test accuracy, test loss, confusion matrix.
        Figure 11 — OOD (when enabled): per-round cosine similarity for every
                    local model vs. the global model, with the OOD threshold.
        Local models (optional): training/val accuracy+loss + confusion matrix.
        """
        os.makedirs("plots", exist_ok=True)
 
        global_id = self.federated_config.host_id
        fed_plot = FederatedPlot()
 
        for model_id in self.models:
            _, _, test_data = self.datasets[model_id]
            title = f"Global Model (id={model_id})" if model_id == global_id \
                    else f"Local Model (id={model_id})"
 
            if model_id == global_id:
                # Figure 10: test accuracy + test loss curves (subplots 1 & 2)
                self.models[model_id].plot_test(xlabel="Round", title=title)
                # Figure 10: confusion matrix on the same figure (subplot 3)
                self.models[model_id].plot.confusion_matrix(
                    self.models[model_id].model, test_data, title
                )
            else:
                # Local models: training history + confusion matrix
                self.models[model_id].plot_all(
                    test_data=test_data, xlabel="Round", title=title
                )
 
            # Save figure to plots/ for inclusion in the report
            plt.figure(num=title)
            plt.savefig(os.path.join("plots", f"{title}.png"),
                        bbox_inches='tight', dpi=150)
            print(f"[Saved] plots/{title}.png")
 
        # Figure 11: OOD similarity plot (only when OOD is enabled)
        if self.ood_config.enabled and self.ood_similarity_results:
            ood_title = "OOD Detection — Similarity vs. Global Model"
            fed_plot.plot_ood_dict(
                result=self.ood_similarity_results,
                federated_config=self.federated_config,
                ood_config=self.ood_config,
                xlabel="Round",
                title=ood_title
            )
            plt.figure(num=ood_title)
            plt.tight_layout()
            plt.savefig(os.path.join("plots", "OOD_Similarity.png"),
                        bbox_inches='tight', dpi=150)
            print("[Saved] plots/OOD_Similarity.png")
 
        input("\nPress Enter to close plots and exit...")
 
    def initialize_models(self):
        # Number of clients (including global)
        num_clients = self.federated_config.clients
        client_to_dataset = self.federated_config.client_to_dataset
 
        # Create a base model to copy from
        base_model = self.init_model
 
        for model_id in range(num_clients):
            # Deep copy the base model for each client (including global)
            self.models[model_id] = copy.deepcopy(base_model)
            # Assign datasets using the mapping
            self.datasets[model_id] = self.dataset.get(client_to_dataset[model_id])
 
    def regression(self, global_id: int):
        """
        3.2.2 Regression: Distribute global model weights to all local models.
 
        At the start of each round the global model's weights are copied into
        every local model so they all begin training from the same shared state.
        This is the 'download' step in standard Federated Learning.
 
        Args:
            global_id (int): ID of the global model (typically 0).
        """
        global_weights = self.models[global_id].model.get_weights()
 
        for model_id in self.models:
            if model_id != global_id:
                self.models[model_id].model.set_weights(global_weights)
 
        if self.federated_config.debug:
            print(f"[Regression] Global model weights (id={global_id}) "
                  f"distributed to {len(self.models) - 1} local model(s).")
 
    def train(self, selected_clients: list, round_idx: int):
        """
        3.2.3 Train: Train each selected local model on its own dataset.
 
        Each local model trains for one round (epoch count is set in model_config).
        The Model.train() method accumulates history across rounds so plots show
        the full training curve, not just the last round.
 
        The global model is NOT trained here — it is evaluated separately in
        run() after aggregation.
 
        Args:
            selected_clients (list): IDs of local models participating this round.
            round_idx (int): Current round index (0-based), used for logging.
        """
        for model_id in selected_clients:
            train_data, val_data, _ = self.datasets[model_id]
 
            if self.federated_config.debug:
                print(f"[Train] Round {round_idx + 1} — training local model id={model_id}")
 
            self.models[model_id].train(train_data, val_data)
 
    def aggregation(self, selected_clients: list, round_idx: int):
        """
        3.2.4 / 3.3.4 Aggregation: FedAvg with optional OOD filtering.
 
        If OOD detection is enabled, each local model's feature bundle is compared
        to the global model's bundle via cosine similarity. Clients below the OOD
        threshold are excluded before FedAvg is applied.
 
        Args:
            selected_clients (list): IDs of local models that trained this round.
            round_idx (int): Current round index (0-based), used for logging.
        """
        # OOD detection: filter out poisoned local models before aggregation
        if self.ood_config.enabled:
            clients_to_aggregate, similarities = self.ood_detection(selected_clients)
 
            # Track per-round similarity scores for plotting (Figure 11)
            for client_id in selected_clients:
                if client_id not in self.ood_similarity_results:
                    self.ood_similarity_results[client_id] = []
                self.ood_similarity_results[client_id].append(
                    similarities.get(client_id, 1.0)
                )
        else:
            clients_to_aggregate = selected_clients
 
        # Safety: if all clients excluded, skip aggregation this round
        if not clients_to_aggregate:
            print(f"[Aggregation] Round {round_idx + 1}: all clients excluded as OOD — skipping.")
            return
 
        # Collect the full weight list from each participating local model.
        all_weights = [self.models[cid].model.get_weights() for cid in clients_to_aggregate]
 
        num_layers = len(all_weights[0])
 
        # Element-wise mean across clients for every layer (FedAvg)
        averaged_weights = [
            federated_math.federated_mean(
                [all_weights[model_i][layer_i] for model_i in range(len(clients_to_aggregate))]
            )
            for layer_i in range(num_layers)
        ]
 
        # Push the averaged weights into the global model.
        global_id = self.federated_config.host_id
        self.models[global_id].model.set_weights(averaged_weights)
 
        if self.federated_config.debug:
            print(f"[Aggregation] Round {round_idx + 1}: averaged weights from "
                  f"{len(clients_to_aggregate)} client(s) → global model (id={global_id})")
 
    def _save_models(self, round_idx: int):
        """Save all model weights to disk after each round.
 
        Weights are saved as Keras H5 files:
            <path>/model_<id>_round_<round>.weights.h5
 
        Args:
            round_idx (int): Current round index (0-based). File uses 1-based round number.
        """
        path = self.federated_config.path
        os.makedirs(path, exist_ok=True)
        round_num = round_idx + 1
        for model_id in self.models:
            filepath = os.path.join(path, f"model_{model_id}_round_{round_num}.weights.h5")
            self.models[model_id].model.save_weights(filepath)
            if self.federated_config.debug:
                print(f"[Save] Model {model_id} → {filepath}")
 
    def _load_models(self):
        """Load saved model weights from disk.
 
        Loads the round specified by federated_config.load_round.
        If a weight file does not exist for a model (e.g., the number of clients
        differs between pre-training and the current experiment), that model keeps
        its randomly-initialized weights and a warning is printed.
        """
        path = self.federated_config.path
        load_round = self.federated_config.load_round
        for model_id in self.models:
            filepath = os.path.join(path, f"model_{model_id}_round_{load_round}.weights.h5")
            if os.path.exists(filepath):
                self.models[model_id].model.load_weights(filepath)
                print(f"[Load] Model {model_id} ← {filepath}")
                if self.federated_config.delete_on_load:
                    os.remove(filepath)
                    print(f"[Load] Deleted {filepath}")
            else:
                print(f"[Load] WARNING: No weights at {filepath} — using initialized weights for model {model_id}")