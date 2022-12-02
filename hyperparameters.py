# Fixed random seed
seed = 11

# The name of the dataset to train/test on
dataset_name = "amazon"

# Values for (epsilon, delta) differential privacy
epsilon, delta = 8, 1e-5

# Number of training epochs
encoder_epochs = 100
model_epochs = 300

# Number of message passing steps to be performed in the aggregation module
num_hops = 5

# Dimensions for the encoder module
encoder_hidden_dims = [300]
encoder_output_dim = 60

# Dimensions for the classification module
classifier_base_hidden_dims = []
classifier_base_output_dim = 20
classifier_head_hidden_dims = []
