import os
import tensorflow as tf
import random
import numpy as np

from dataset.dataset import Dataset
from config import ConfigFederated, ConfigOod, ConfigModel, ConfigDataset, ConfigPlot
from dataset.download.a_faces16000 import Afaces16000
from dataset.download.b_alzheimer5100 import Balzheimer5100
from dataset.download.b_alzheimer5100_poisoned import Balzheimer5100_poisoned
from dataset.download.b_tumor3000 import Btumor3000
from dataset.download.b_tumor4600 import Btumor4600
from dataset.download.l_pneumonia5200 import Lpneumonia5200
from federated.federated import Federated
from model.model import Model

from model.math.plot import ModelPlot

#
# FEEL FREE TO EDIT THE CONTENT OF ALL GIVEN FILES AS YOU LIKE.
#

############# REPRODUCIBILITY, deterministic behavior #############
def set_seeds(SEED):
    """ Set seeds for deterministic behavior. 
    """
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

def set_global_determinism(SEED):
    set_seeds(SEED=SEED)
    
    # Uncomment below if limiting cpu treads, may help with determinism. However may slow down training, especially on large models.
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # CUDA/GPU users
    # os.environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # tf.config.experimental.enable_op_determinism()

SEED = 42   # Change seed as you like. 
set_global_determinism(SEED=SEED)
###############################################################

class ModelSimulation():
    """ Only for testing single model. 
    """
           
    def run(self):
        #-----------CONFIG--------------------
        model_config = ConfigModel(
            debug = True,
            epochs = 5,
            activation = 'relu',
            activation_out = 'softmax',
            optimizer = 'adam',
            loss = 'categorical_crossentropy'
        )
        dataset_config = ConfigDataset(
            debug = True,                   # DISABLE IF YOU WANT TO PREVENT IMAGE EXAMPLES FROM BEING DISPLAYED BEFORE TRAINING.
            batch_size = 64,                          
            image_size = 256,            
            input_shape = (256,256,1),   
            split = 0.25,
            number_of_classes=2
        )
        plot_config = ConfigPlot(
            plot = True,
            path = "./.env/.saved/",
            img_per_class = 10  
        )
        
        #-----------SIM---------------------- 
        m = Model(
            model_config=model_config,
            dataset_config=dataset_config,
            plot_config=plot_config
        )
        
        # Create dataset by merging multiple datasets together. 
        train_data, validation_data, test_data = Dataset(
            [
                (Btumor4600().ID, Btumor4600(), []),
                (Btumor3000().ID, Btumor3000(), []),
                (Balzheimer5100().ID, Balzheimer5100(), []),
                (Lpneumonia5200().ID, Lpneumonia5200(), [])
                
            ],
            dataset_config=dataset_config,
            plot_config=plot_config
        ).mergeAll()
        
        # Below is an example of dataset with subsets. That can be used in federated learning context. 
        # All given datasets down below are available in ./dataset/download.
        #
        # dataset = Dataset(
        #     [
        #         (Btumor4600().ID, Btumor4600(), []),    # id 0 ID DATA 
        #         (Btumor3000().ID, Btumor3000(), []),    # id 1 ID DATA 
        #         (Balzheimer5100().ID, Balzheimer5100(), []), # id 2 ID DATA 
        #         (Lpneumonia5200().ID, Lpneumonia5200(), []), # id 3 ID DATA 
        #         (Lpneumonia5200().ID, Lpneumonia5200(), [[300,500],[3600,3800]]), # id 4 ID DATA (subsamples of total, not used in pre-training) 
        #         (Btumor4600().ID, Btumor4600(), [[300,500],[3700,3900]]), # id 5 ID DATA (subsamples of total, not used in pre-training)  
        #         (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), [[1000,1700],[4000,4700]]), # id 6 OOD DATA (poisoned data) (not used in pre-training)
        #         (Afaces16000().ID, Afaces16000(), [[1,700],[2501,3200]]) # id 7 OOD DATA (not used in pre-training) # Take some two subsets of complete dataset. # [250,750], [4000,4500]
        #     ],
        #     dataset_config=dataset_config,
        #     plot_config=plot_config
        # )
        #
        # Bind local model / client to dataset id / subsets.
        #
        # train_data, validation_data, test_data = dataset.get(index i)
        #
        
        m.train(train_data, validation_data)     
        m.test(test_data) 
        
        m.plot_all(test_data, xlabel="Epoch", title="CNN Model")
        input('Press Enter to close plots and exit...')
        
class FederatedSimulation():
    """
        TODO:
        Simulation for federated learning with multiple local models.
        You are free to change the configuration as you like, remove or edit.
        
        This are only some parameters for simulation that might be of interest.
        
        You may remove the configuration entirely if you prefer and apply an alternative setup.
    """
    #--------------------------------------- 
    #--------------CONFIG-------------------
    #--------------------------------------- 
    federated_config = ConfigFederated(
        debug = True,
            
        # _________FILE____________
        save = False,                         # IF save global model after sim. 
        load_round = 0,                       # Which specific global model from directory (directory id).
        load_reg = True,                      # Regression from global model after loading pretrained or not.
        load = False,                         # IF load from directory.
        delete_on_load = False,               # Delete loaded model in directory.
        path = "./.env/.saved/",              # Path to saved global model.
            
        # _______SIMULATION________
        rounds = 25,                          # Number of rounds for sim in federated. 
        ood_round = 26,                       # Round where ood starts.  
        clients = 5,                          # Number of clients in sim. (global model + local models).
        participants = 4,                     # How many participants for current round (randomized between clients). (local models only)
        host_id=0,                            # Host id (index). Should probably always be 0 (global model).
        client_to_dataset=[[0,1,2,3],[0],[1],[2],[3]]   # local model -> dataset assignment (array index in dataset).
    )
    
    ood_config = ConfigOod(
        debug = True,
        hdc_debug = False,
            
        # _______SIMULATION________
        enabled = False,                            # Enabling hdff and ood. 
        hyper_size=int(1e4),                        # Hyper dimensions for projection matrix. 
        
        id_client = [1,2,3,4],                      # Clients id in-distribution, excluding global client/model.
        ood_client = [5,6],                           # Datasets out-of-distribution (index in dataset), excluding global client/model.
        
        ood_protection = True,                     # IF ood protection (exluding) is enabled.
        ood_protection_thres = 0.7                  # Threshold for consider models being ood. 
    )
    
    model_config = ConfigModel(
        debug = True,
            
        # _______SIMULATION________
        epochs = 1,                         # Highly recommend to be 1 for federated. Change rounds instead. 
        activation = 'relu',                # Parameters for model.
        activation_out = 'softmax',
        optimizer = 'adam',
        loss = 'categorical_crossentropy'   # 'categorical_crossentropy'
    )
    dataset_config = ConfigDataset(
        debug = False,
            
        # _______SIMULATION________
        batch_size = 64,                    # Batchsize for datasets.                          
        image_size = 256,                   # Image size 
        input_shape = (256,256,1),
        split = 0.25,                       # Train / validation split.
        number_of_classes = 2               # Number of total classes.
    )
    plot_config = ConfigPlot(
        plot = False,
            
        # __________FILE___________
        path = './.env/plot',                
        img_per_class = 10                  # Only for plotting example pictures from dataset (debug must be set to true).
    )
    
    def run(self):
        #------------------------------------ 
        #--------------SIM------------------- 
        #------------------------------------ 
        m = Model(
            model_config=self.model_config,
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        # client-to-dataset and id dataset and ood dataset are referenced to this list. 
        dataset = Dataset(
            [
                (Btumor4600().ID, Btumor4600(), []),    # id 0 ID DATA 
                (Btumor3000().ID, Btumor3000(), []),    # id 1 ID DATA 
                (Balzheimer5100().ID, Balzheimer5100(), []), # id 2 ID DATA 
                (Lpneumonia5200().ID, Lpneumonia5200(), []), # id 3 ID DATA 
                (Lpneumonia5200().ID, Lpneumonia5200(), [[300,500],[3600,3800]]), # id 4 ID DATA (subsamples of total, not used in pre-training) 
                (Btumor4600().ID, Btumor4600(), [[300,500],[3700,3900]]), # id 5 ID DATA (subsamples of total, not used in pre-training) 
                (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), [[1000,1700],[4000,4700]]), # id 6 OOD DATA (poisoned data) (not used in pre-training)
                (Afaces16000().ID, Afaces16000(), [[1,700],[2501,3200]]) # id 7 OOD DATA (not used in pre-training) # Take some two subsets of complete dataset. # [250,750], [4000,4500]
            ],
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        federated = Federated(
            dataset=dataset, 
            model=m,
            federated_config=self.federated_config,
            ood_config=self.ood_config, 
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        return federated.run()
    
class PreTrainingSimulation():
    """
    Phase 1: Pre-train all models on all ID data for 35 rounds.
    OOD detection is disabled. Models are saved after each round to disk.
    Run this FIRST before any experiment.
    """
    SAVE_PATH = "./.env/.saved/"
    ROUNDS    = 35

    federated_config = ConfigFederated(
        debug          = True,
        save           = True,                       # Save weights after every round
        load_round     = 0,
        load_reg       = True,
        load           = False,
        delete_on_load = False,
        path           = "./.env/.saved/",
        rounds         = ROUNDS,
        ood_round      = ROUNDS + 1,                 # OOD never triggered
        clients        = 5,                          # global(0) + 4 local
        participants   = 4,
        host_id        = 0,
        client_to_dataset = [[0,1,2,3],[0],[1],[2],[3]]
    )
    ood_config = ConfigOod(
        debug               = False,
        hdc_debug           = False,
        enabled             = False,                 # OOD disabled during pre-training
        hyper_size          = int(1e4),
        id_client           = [1,2,3,4],
        ood_client          = [],
        ood_protection      = False,
        ood_protection_thres= 0.7
    )
    model_config = ConfigModel(
        debug          = False,
        epochs         = 1,
        activation     = 'relu',
        activation_out = 'softmax',
        optimizer      = 'adam',
        loss           = 'categorical_crossentropy'
    )
    dataset_config = ConfigDataset(
        debug           = False,
        batch_size      = 64,
        image_size      = 256,
        input_shape     = (256,256,1),
        split           = 0.25,
        number_of_classes = 2
    )
    plot_config = ConfigPlot(plot=False, path='./.env/plot', img_per_class=10)

    def run(self):
        m = Model(
            model_config   = self.model_config,
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
        dataset = Dataset(
            [
                (Btumor4600().ID,    Btumor4600(),    []),  # id 0
                (Btumor3000().ID,    Btumor3000(),    []),  # id 1
                (Balzheimer5100().ID, Balzheimer5100(), []),# id 2
                (Lpneumonia5200().ID, Lpneumonia5200(), []),# id 3
            ],
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
        federated = Federated(
            dataset          = dataset,
            model            = m,
            federated_config = self.federated_config,
            ood_config       = self.ood_config,
            dataset_config   = self.dataset_config,
            plot_config      = self.plot_config
        )
        return federated.run()


class Experiment1Simulation():
    """
    3.4.1 Experiment 1: 1 OOD local model with complete poisoned dataset.
    OOD detection DISABLED.

    Setup:
      - Global model (id=0): loads pre-trained weights, tests on all 4 ID datasets.
      - Local model  (id=1): starts from global weights (load_reg=True),
                             trains on complete Balzheimer5100_poisoned().
    Run for 5 rounds and observe deterioration in global model accuracy.

    IMPORTANT: Run PreTrainingSimulation first to generate saved weights.
    """
    LOAD_ROUND = 35          # Must match PreTrainingSimulation.ROUNDS
    SAVE_PATH  = "./.env/.saved/"

    federated_config = ConfigFederated(
        debug          = True,
        save           = False,
        load_round     = LOAD_ROUND,
        load_reg       = True,                       # Sync local from global after load
        load           = True,                       # Load pre-trained global model
        delete_on_load = False,
        path           = SAVE_PATH,
        rounds         = 5,
        ood_round      = 6,                          # OOD never triggered
        clients        = 2,                          # global(0) + 1 OOD local(1)
        participants   = 1,
        host_id        = 0,
        client_to_dataset = [[0,1,2,3],[4]]          # global: all 4 ID sets; local: poisoned
    )
    ood_config = ConfigOod(
        debug               = True,
        hdc_debug           = False,
        enabled             = False,                 # OOD detection DISABLED for Experiment 1
        hyper_size          = int(1e4),
        id_client           = [1],
        ood_client          = [1],
        ood_protection      = False,
        ood_protection_thres= 0.7
    )
    model_config = ConfigModel(
        debug          = True,
        epochs         = 1,
        activation     = 'relu',
        activation_out = 'softmax',
        optimizer      = 'adam',
        loss           = 'categorical_crossentropy'
    )
    dataset_config = ConfigDataset(
        debug           = False,
        batch_size      = 64,
        image_size      = 256,
        input_shape     = (256,256,1),
        split           = 0.25,
        number_of_classes = 2
    )
    plot_config = ConfigPlot(plot=False, path='./.env/plot', img_per_class=10)

    def run(self):
        m = Model(
            model_config   = self.model_config,
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
        dataset = Dataset(
            [
                (Btumor4600().ID,    Btumor4600(),    []),               # id 0  (ID)
                (Btumor3000().ID,    Btumor3000(),    []),               # id 1  (ID)
                (Balzheimer5100().ID, Balzheimer5100(), []),             # id 2  (ID)
                (Lpneumonia5200().ID, Lpneumonia5200(), []),             # id 3  (ID)
                (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), []),  # id 4 (OOD — complete poisoned)
            ],
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
        federated = Federated(
            dataset          = dataset,
            model            = m,
            federated_config = self.federated_config,
            ood_config       = self.ood_config,
            dataset_config   = self.dataset_config,
            plot_config      = self.plot_config
        )
        return federated.run()

class Experiment2Simulation():
    """
    3.4.2 Experiment 2: 1 OOD local model with complete poisoned dataset.
    OOD detection ENABLED.
 
    Setup:
      - Global model (id=0): loads pre-trained weights, tests on all 4 ID datasets.
      - Local model  (id=1): starts from global weights (load_reg=True),
                             trains on complete Balzheimer5100_poisoned().
      - OOD detection is active from round 1. The HDFF mechanism compares the
        local model's feature bundle to the global model's. If cosine similarity
        drops below ood_protection_thres (0.7), client 1 is excluded from FedAvg.
 
    Expected result: OOD detection identifies the poisoned client and the global
    model accuracy remains stable (protected) across all 5 rounds.
 
    IMPORTANT: Run PreTrainingSimulation first to generate saved weights.
    """
    LOAD_ROUND = 35
    SAVE_PATH  = "./.env/.saved/"
 
    federated_config = ConfigFederated(
        debug          = True,
        save           = False,
        load_round     = LOAD_ROUND,
        load_reg       = True,                       # Sync local from global after load
        load           = True,                       # Load pre-trained global model
        delete_on_load = False,
        path           = SAVE_PATH,
        rounds         = 5,
        ood_round      = 1,                          # OOD detection active from round 1
        clients        = 2,                          # global(0) + 1 OOD local(1)
        participants   = 1,
        host_id        = 0,
        client_to_dataset = [[0,1,2,3],[4]]          # global: all 4 ID sets; local: complete poisoned
    )
    ood_config = ConfigOod(
        debug               = True,
        hdc_debug           = False,
        enabled             = True,                  # OOD detection ENABLED for Experiment 2
        hyper_size          = int(1e4),
        id_client           = [],                    # No clean local clients in this setup
        ood_client          = [1],                   # Client 1 is the OOD (poisoned) client
        ood_protection      = True,                  # Exclude detected OOD clients from FedAvg
        ood_protection_thres= 0.7                    # Similarity < 0.7 → client is OOD
    )
    model_config = ConfigModel(
        debug          = True,
        epochs         = 1,
        activation     = 'relu',
        activation_out = 'softmax',
        optimizer      = 'adam',
        loss           = 'categorical_crossentropy'
    )
    dataset_config = ConfigDataset(
        debug           = False,
        batch_size      = 64,
        image_size      = 256,
        input_shape     = (256,256,1),
        split           = 0.25,
        number_of_classes = 2
    )
    plot_config = ConfigPlot(plot=False, path='./.env/plot', img_per_class=10)
 
    def run(self):
        m = Model(
            model_config   = self.model_config,
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
        dataset = Dataset(
            [
                (Btumor4600().ID,    Btumor4600(),    []),               # id 0  (ID)
                (Btumor3000().ID,    Btumor3000(),    []),               # id 1  (ID)
                (Balzheimer5100().ID, Balzheimer5100(), []),             # id 2  (ID)
                (Lpneumonia5200().ID, Lpneumonia5200(), []),             # id 3  (ID)
                (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), []),  # id 4 (OOD — complete poisoned)
            ],
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
        federated = Federated(
            dataset          = dataset,
            model            = m,
            federated_config = self.federated_config,
            ood_config       = self.ood_config,
            dataset_config   = self.dataset_config,
            plot_config      = self.plot_config
        )
        return federated.run()
 
 
class Experiment3Simulation():
    """
    3.4.3 Experiment 3: 4 ID local models + 1 OOD local model with complete poisoned dataset.
    OOD detection ENABLED.
 
    Setup:
      - Global model (id=0): loads pre-trained weights, tests on all 4 ID datasets.
      - Local model  (id=1): Btumor4600()        — full ID dataset
      - Local model  (id=2): Btumor3000()        — full ID dataset
      - Local model  (id=3): Balzheimer5100()    — full ID dataset
      - Local model  (id=4): Lpneumonia5200()    — full ID dataset
      - Local model  (id=5): Balzheimer5100_poisoned() — complete label-flipped OOD dataset
 
    OOD detection is active from round 1. The HDFF mechanism compares each local
    model's feature bundle to the global model's. If cosine similarity drops below
    ood_protection_thres (0.7), that client is excluded from FedAvg.
 
    Expected result:
      - Local model id=5 is identified as OOD every round (similarity << 0.7).
      - The 4 clean local models contribute normally to FedAvg.
      - Global model accuracy remains stable or improves across all 5 rounds.
 
    IMPORTANT: Run PreTrainingSimulation first to generate saved weights.
    """
    LOAD_ROUND = 35
    SAVE_PATH  = "./.env/.saved/"
 
    federated_config = ConfigFederated(
        debug          = True,
        save           = False,
        load_round     = LOAD_ROUND,
        load_reg       = True,                        # Sync all locals from global after load
        load           = True,                        # Load pre-trained global model
        delete_on_load = False,
        path           = SAVE_PATH,
        rounds         = 5,
        ood_round      = 1,                           # OOD detection active from round 1
        clients        = 6,                           # global(0) + 4 ID locals + 1 OOD local
        participants   = 5,                           # All 5 local clients participate each round
        host_id        = 0,
        client_to_dataset = [[0,1,2,3],[0],[1],[2],[3],[4]]
        # global: all 4 ID datasets for evaluation
        # local 1 → Btumor4600 (dataset idx 0)
        # local 2 → Btumor3000 (dataset idx 1)
        # local 3 → Balzheimer5100 (dataset idx 2)
        # local 4 → Lpneumonia5200 (dataset idx 3)
        # local 5 → Balzheimer5100_poisoned (dataset idx 4)  ← OOD
    )
    ood_config = ConfigOod(
        debug               = True,
        hdc_debug           = False,
        enabled             = True,                   # OOD detection ENABLED
        hyper_size          = int(1e4),
        id_client           = [1, 2, 3, 4],           # Clean local clients
        ood_client          = [5],                    # Poisoned OOD client
        ood_protection      = True,                   # Exclude detected OOD clients from FedAvg
        ood_protection_thres= 0.7                     # Similarity < 0.7 → client is OOD
    )
    model_config = ConfigModel(
        debug          = True,
        epochs         = 1,
        activation     = 'relu',
        activation_out = 'softmax',
        optimizer      = 'adam',
        loss           = 'categorical_crossentropy'
    )
    dataset_config = ConfigDataset(
        debug           = False,
        batch_size      = 64,
        image_size      = 256,
        input_shape     = (256,256,1),
        split           = 0.25,
        number_of_classes = 2
    )
    plot_config = ConfigPlot(plot=False, path='./.env/plot', img_per_class=10)
 
    def run(self):
        m = Model(
            model_config   = self.model_config,
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
        dataset = Dataset(
            [
                (Btumor4600().ID,    Btumor4600(),    []),                        # idx 0  (ID)
                (Btumor3000().ID,    Btumor3000(),    []),                        # idx 1  (ID)
                (Balzheimer5100().ID, Balzheimer5100(), []),                      # idx 2  (ID)
                (Lpneumonia5200().ID, Lpneumonia5200(), []),                      # idx 3  (ID)
                (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), []),    # idx 4  (OOD)
            ],
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
        federated = Federated(
            dataset          = dataset,
            model            = m,
            federated_config = self.federated_config,
            ood_config       = self.ood_config,
            dataset_config   = self.dataset_config,
            plot_config      = self.plot_config
        )
        return federated.run()
 
 
class MixedOodDataset5:
    """
    Combined dataset for Experiment 4 local model (id=5).
 
    Merges:
      - Complete Balzheimer5100_poisoned (all 4 label-flipped directories)
      - Lpneumonia5200 (NORMAL and PNEUMONIA directories appended after)
 
    When used with dynamic indices [[0, balz_total], [balz_total, balz_total+750]],
    the DataframeGenerator selects:
      All Balzheimer5100_poisoned images  (~5121, complete poisoned OOD data)
      First 750 Lpneumonia5200/NORMAL images (ID data, within 500–1000 range)
 
    The directory scan order in DataframeGenerator is:
      [0]  VeryMildDemented  → Healthy  (Balzheimer5100_poisoned)
      [1]  MildDemented      → Healthy  (Balzheimer5100_poisoned)
      [2]  ModerateDemented  → Healthy  (Balzheimer5100_poisoned)
      [3]  NonDemented       → Sick     (Balzheimer5100_poisoned)
      [4]  NORMAL            → Healthy  (Lpneumonia5200)
      [5]  PNEUMONIA         → Sick     (Lpneumonia5200)
    """
    def __init__(self):
        poisoned  = Balzheimer5100_poisoned()
        pneumonia = Lpneumonia5200()
        self.ID     = "MixedOodDataset5"
        # Balzheimer dirs come first, then Lpneumonia dirs
        self.paths  = poisoned.paths  + pneumonia.paths
        self.labels = poisoned.labels + pneumonia.labels
 
    def pre_processing(self, image):
        return image
 
 
class Experiment4Simulation():
    """
    3.4.4 Experiment 4: 4 ID local models + 1 mixed OOD local model.
    OOD detection ENABLED.
 
    Setup:
      - Global model (id=0): loads pre-trained weights, tests on all 4 ID datasets.
      - Local model  (id=1): Btumor4600()         — full ID dataset
      - Local model  (id=2): Btumor3000()         — full ID dataset
      - Local model  (id=3): Balzheimer5100()     — full ID dataset
      - Local model  (id=4): Lpneumonia5200()     — full ID dataset
      - Local model  (id=5): MixedOodDataset5()   — complete label-flipped
                             Balzheimer5100_poisoned  +  750 samples from
                             Lpneumonia5200 (500-1000 ID samples, OOD mix)
 
    The mixed OOD client trains on BOTH poisoned (label-flipped) and legitimate
    ID data in the same round.  OOD detection (HDFF cosine-similarity) should
    still identify this client as OOD (similarity < 0.7) because the dominant
    signal from the flipped labels corrupts its feature representation.
 
    IMPORTANT: Run PreTrainingSimulation first to generate saved weights.
    """
    LOAD_ROUND = 35
    SAVE_PATH  = "./.env/.saved/"
 
    federated_config = ConfigFederated(
        debug          = True,
        save           = False,
        load_round     = LOAD_ROUND,
        load_reg       = True,
        load           = True,
        delete_on_load = False,
        path           = SAVE_PATH,
        rounds         = 5,
        ood_round      = 1,                           # OOD detection active from round 1
        clients        = 6,                           # global(0) + 4 ID locals + 1 mixed OOD
        participants   = 5,                           # All 5 local clients participate each round
        host_id        = 0,
        client_to_dataset = [[0,1,2,3],[0],[1],[2],[3],[4]]
        # global: all 4 ID datasets for evaluation
        # local 1 → Btumor4600          (dataset idx 0)
        # local 2 → Btumor3000          (dataset idx 1)
        # local 3 → Balzheimer5100      (dataset idx 2)
        # local 4 → Lpneumonia5200      (dataset idx 3)
        # local 5 → MixedOodDataset5    (dataset idx 4)  ← mixed OOD
    )
    ood_config = ConfigOod(
        debug               = True,
        hdc_debug           = False,
        enabled             = True,                   # OOD detection ENABLED
        hyper_size          = int(1e4),
        id_client           = [1, 2, 3, 4],           # Clean local clients
        ood_client          = [5],                    # Mixed OOD client
        ood_protection      = True,                   # Exclude detected OOD clients from FedAvg
        ood_protection_thres= 0.7                     # Similarity < 0.7 → client is OOD
    )
    model_config = ConfigModel(
        debug          = True,
        epochs         = 1,
        activation     = 'relu',
        activation_out = 'softmax',
        optimizer      = 'adam',
        loss           = 'categorical_crossentropy'
    )
    dataset_config = ConfigDataset(
        debug           = False,
        batch_size      = 64,
        image_size      = 256,
        input_shape     = (256,256,1),
        split           = 0.25,
        number_of_classes = 2
    )
    plot_config = ConfigPlot(plot=False, path='./.env/plot', img_per_class=10)
 
    def run(self):
        import os
 
        # ── Build mixed OOD dataset for local model 5 ───────────────────────
        # MixedOodDataset5 stores Balzheimer5100_poisoned dirs first (indices 0..3)
        # then Lpneumonia5200 dirs (indices 4..5).  We count the exact number of
        # files in the Balzheimer portion so the index ranges precisely select:
        #   ALL  Balzheimer5100_poisoned images   (complete poisoned data)
        #   750  Lpneumonia5200/NORMAL   images   (ID data, in 500–1000 range)
        mixed_ds = MixedOodDataset5()
 
        balz_total = sum(
            len([f for f in os.listdir(d)
                 if os.path.isfile(os.path.join(d, f))])
            for dir_list in mixed_ds.paths[:4]   # first 4 entries = Balzheimer5100_poisoned
            for d in dir_list
        )
        PNEUMONIA_SUBSET = 750   # 500–1000 ID samples from Lpneumonia5200/NORMAL
        mixed_indices = [
            [0,            balz_total],                      # all Balzheimer5100_poisoned
            [balz_total,   balz_total + PNEUMONIA_SUBSET],   # 750 Lpneumonia5200 NORMAL
        ]
        print(f"[Exp4] Balzheimer5100_poisoned file count : {balz_total}")
        print(f"[Exp4] Mixed OOD dataset indices          : {mixed_indices}")
 
        # ── Model & datasets ────────────────────────────────────────────────
        m = Model(
            model_config   = self.model_config,
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
        dataset = Dataset(
            [
                (Btumor4600().ID,    Btumor4600(),    []),              # idx 0  (ID)
                (Btumor3000().ID,    Btumor3000(),    []),              # idx 1  (ID)
                (Balzheimer5100().ID, Balzheimer5100(), []),            # idx 2  (ID)
                (Lpneumonia5200().ID, Lpneumonia5200(), []),            # idx 3  (ID)
                (mixed_ds.ID, mixed_ds, mixed_indices),                # idx 4  (OOD: mixed)
            ],
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
 
        # ── Federated simulation ─────────────────────────────────────────────
        federated = Federated(
            dataset          = dataset,
            model            = m,
            federated_config = self.federated_config,
            ood_config       = self.ood_config,
            dataset_config   = self.dataset_config,
            plot_config      = self.plot_config
        )
        return federated.run()
 
 
class Experiment5Simulation():
    """
    3.4.5 Experiment 5: 4 ID local models + 1 new OOD local model (Afaces16000).
    OOD detection ENABLED.
 
    Setup:
      - Global model (id=0): loads pre-trained weights, tests on all 4 ID datasets.
      - Local model  (id=1): Btumor4600()    — full ID dataset (brain tumour MRI)
      - Local model  (id=2): Btumor3000()    — full ID dataset (brain tumour MRI)
      - Local model  (id=3): Balzheimer5100()— full ID dataset (Alzheimer MRI)
      - Local model  (id=4): Lpneumonia5200()— full ID dataset (chest X-ray)
      - Local model  (id=5): Afaces16000()   — TRUE OOD: animal face photographs
                             (cat → Healthy, dog → Sick)
 
    Afaces16000 is the most extreme OOD scenario: colour photographs of animal
    faces are completely outside the medical-imaging domain that the global model
    was pre-trained on.  No label flipping is needed — the feature distribution
    itself is radically different.
 
    Expected outcome:
      - Local model 5 should produce the LOWEST cosine similarity scores of all
        experiments (feature bundles dominated by RGB animal-texture features vs
        the global model's grayscale medical-imaging features).
      - OOD detection trivially rejects model 5 every round (similarity << 0.7).
      - Global model accuracy remains stable / improves across 5 rounds.
 
    IMPORTANT: Run PreTrainingSimulation first to generate saved weights.
    """
    LOAD_ROUND = 35
    SAVE_PATH  = "./.env/.saved/"
 
    federated_config = ConfigFederated(
        debug          = True,
        save           = False,
        load_round     = LOAD_ROUND,
        load_reg       = True,
        load           = True,
        delete_on_load = False,
        path           = SAVE_PATH,
        rounds         = 5,
        ood_round      = 1,                           # OOD detection active from round 1
        clients        = 6,                           # global(0) + 4 ID locals + 1 true OOD
        participants   = 5,                           # All 5 local clients participate each round
        host_id        = 0,
        client_to_dataset = [[0,1,2,3],[0],[1],[2],[3],[4]]
        # global: all 4 ID datasets for evaluation
        # local 1 → Btumor4600      (dataset idx 0)
        # local 2 → Btumor3000      (dataset idx 1)
        # local 3 → Balzheimer5100  (dataset idx 2)
        # local 4 → Lpneumonia5200  (dataset idx 3)
        # local 5 → Afaces16000     (dataset idx 4)  ← true OOD (animal faces)
    )
    ood_config = ConfigOod(
        debug               = True,
        hdc_debug           = False,
        enabled             = True,                   # OOD detection ENABLED
        hyper_size          = int(1e4),
        id_client           = [1, 2, 3, 4],           # Clean local clients
        ood_client          = [5],                    # True OOD client (animal faces)
        ood_protection      = True,                   # Exclude detected OOD clients from FedAvg
        ood_protection_thres= 0.7                     # Similarity < 0.7 → client is OOD
    )
    model_config = ConfigModel(
        debug          = True,
        epochs         = 1,
        activation     = 'relu',
        activation_out = 'softmax',
        optimizer      = 'adam',
        loss           = 'categorical_crossentropy'
    )
    dataset_config = ConfigDataset(
        debug           = False,
        batch_size      = 64,
        image_size      = 256,
        input_shape     = (256,256,1),
        split           = 0.25,
        number_of_classes = 2
    )
    plot_config = ConfigPlot(plot=False, path='./.env/plot', img_per_class=10)
 
    def run(self):
        m = Model(
            model_config   = self.model_config,
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
        dataset = Dataset(
            [
                (Btumor4600().ID,    Btumor4600(),    []),              # idx 0  (ID)
                (Btumor3000().ID,    Btumor3000(),    []),              # idx 1  (ID)
                (Balzheimer5100().ID, Balzheimer5100(), []),            # idx 2  (ID)
                (Lpneumonia5200().ID, Lpneumonia5200(), []),            # idx 3  (ID)
                (Afaces16000().ID,   Afaces16000(),   []),              # idx 4  (TRUE OOD)
            ],
            dataset_config = self.dataset_config,
            plot_config    = self.plot_config
        )
        federated = Federated(
            dataset          = dataset,
            model            = m,
            federated_config = self.federated_config,
            ood_config       = self.ood_config,
            dataset_config   = self.dataset_config,
            plot_config      = self.plot_config
        )
        return federated.run()
 
 
if __name__ == "__main__":
    # ── Step 1: Pre-train (run once, then comment out) ──
    # sim_pretrain = PreTrainingSimulation()
    # sim_pretrain.run()
 
    # ── Step 2: Experiment 1 — 1 OOD local model, OOD detection disabled ──
    # sim_exp1 = Experiment1Simulation()
    # sim_exp1.run()
 
    # ── Step 3: Experiment 2 — 1 OOD local model, OOD detection ENABLED ──
    # sim_exp2 = Experiment2Simulation()
    # sim_exp2.run()
 
    # ── Step 4: Experiment 3 — 4 ID locals + 1 OOD local, OOD detection ENABLED ─
    # sim_exp3 = Experiment3Simulation()
    # sim_exp3.run()
 
    # ── Step 5: Experiment 4 — 4 ID locals + 1 MIXED OOD local, OOD detection ENABLED ─
    # sim_exp4 = Experiment4Simulation()
    # sim_exp4.run()
 
    # ── Step 6: Experiment 5 — 4 ID locals + 1 TRUE OOD local (Afaces16000), OOD detection ENABLED ─
    sim_exp5 = Experiment5Simulation()
    sim_exp5.run()