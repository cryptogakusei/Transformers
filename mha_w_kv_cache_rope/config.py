import urllib.request 


# Model configuration
MODEL_CONFIG = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,
    "num_heads" : 12,
    "num_layers" : 12,
    "dropout_rate" : 0.1,
    "qkv_bias" : False,
    "rope_limit" : 4096,
    "kvcache_limit" : 4096,
 }

# Training configuration
TRAINING_CONFIG = {
    "dataset" : "roneneldan/TinyStories",
    "train_ratio" : 0.95,
    "num_epochs" : 1,
    "batch_size" : 16,
    "eval_freq" : 500, # how frequently evaluation to be run
    "eval_num_batches" : 20, # number of batches to run evaluation over at each go
}


# TRAINING_CONFIG = {
#     "url" : "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
#     "train_ratio" : 0.9,
#     "num_epochs" : 10,
#     "batch_size" : 2,
#     "eval_freq" : 5, # how frequently evaluation to be run
#     "eval_num_batches" : 5, # number of batches to run evaluation over at each go
# }

