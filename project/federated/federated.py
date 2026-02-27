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
        self.initialize_models()
    
    def run(self): 
        """
            Running federated learning environment. 
        """
        for round_idx in range(self.federated_config.rounds):
            print(f"\n--- Round {round_idx + 1}/{self.federated_config.rounds} ---")
            
            # 1. Regression (Distribute global weights to local models)
            # Assuming global model has ID 0
            self.regression(0)
            
            # 2. Select clients for this round
            # Typically excludes global model (ID 0)
            selected_clients = list(range(1, self.federated_config.clients))
            
            # 3. Train selected local models
            self.train(selected_clients, round_idx)
            
            # 4. Aggregation (FedAvg)
            self.aggregation(selected_clients, round_idx)
            
            # 5. Evaluate Global Model
            # Global model (ID 0) only performs testing on all ID data
            self.models[0].test(self.datasets[0][2]) # test_data is the 3rd element from generator

        # 6. Plot results for all models after all rounds complete
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
        """ Exctract features from model. 

        Args:
            id (int): Id of model.
            model (Model): Model (object).
        """
        # TODO
        
        return None
        
    def ood_detection(self, selected_clients):
        """ Detecting model being ood from selected clients that trains.

        Args:
            selected_clients (int): clients that undergo training this round.
        """
        # TODO
        
        return None
            
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

        # Figure 11: OOD similarity plot (requires OOD to be enabled and
        # self.ood_similarity_results populated during aggregation rounds)
        if self.ood_config.enabled and hasattr(self, 'ood_similarity_results'):
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
        3.2.4 Aggregation (FedAvg): Average local model weights → update global model.

        For every layer in the network, the weights from all participating local
        models are collected and averaged using federated_mean(). The result is
        written back into the global model, completing the FedAvg round.

        Args:
            selected_clients (list): IDs of local models that trained this round.
            round_idx (int): Current round index (0-based), used for logging.
        """
        # Collect the full weight list from each participating local model.
        # all_weights[i] is a list of numpy arrays — one array per layer.
        all_weights = [self.models[cid].model.get_weights() for cid in selected_clients]

        num_layers = len(all_weights[0])

        # For each layer, gather that layer's weight array from every local model
        # and compute the element-wise mean via federated_mean().
        averaged_weights = [
            federated_math.federated_mean(
                [all_weights[model_i][layer_i] for model_i in range(len(selected_clients))]
            )
            for layer_i in range(num_layers)
        ]

        # Push the averaged weights into the global model.
        global_id = self.federated_config.host_id
        self.models[global_id].model.set_weights(averaged_weights)

        if self.federated_config.debug:
            print(f"[Aggregation] Round {round_idx + 1}: averaged weights from "
                  f"{len(selected_clients)} client(s) → global model (id={global_id})")