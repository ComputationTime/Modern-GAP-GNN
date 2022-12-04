import torch

class config:
    ################################################################################
    # Pytorch Settings                                                             #
    ################################################################################

    # Fixed random seed
    seed = 11

    # Whether or not to use CPU or GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ################################################################################
    # Training Settings                                                            #
    ################################################################################

    # The name of the dataset to train/test on
    dataset_name = "amazon"

    # The aggregation module to use (pma, pmat, pmwa)
    aggregation_module_name = "pmwa"

    # Values for (epsilon, delta) differential privacy
    epsilon, delta, alpha = 8, 1e-5, 2

    # Number of training iterations
    encoder_training_iters = 1000
    pmat_training_iters = 500
    model_training_iters = 1000

    # Size of each batch to use during training
    batch_size = 256

    # Number of message passing steps to be performed in the aggregation module
    num_hops = 1

    # Dimensions for the encoder module
    encoder_hidden_dims = [300]
    encoder_output_dim = 60

    # Dimensions for the classification module
    classifier_base_hidden_dims = []
    classifier_base_output_dim = 20
    classifier_head_hidden_dims = []
