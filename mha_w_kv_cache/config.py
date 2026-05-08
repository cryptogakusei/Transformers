import urllib.request 


# Model configuration
MODEL_CONFIG = {
    "vocab_size" : 50257,
    "context_length" : 256,
    "emb_dim" : 768,
    "num_heads" : 12,
    "num_layers" : 12,
    "dropout_rate" : 0.1,
    "qkv_bias" : False,
}

# Training configuration
TRAINING_CONFIG = {
    # "url" : "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt",
    "dataset" : "roneneldan/TinyStories",
    "train_ratio" : 0.95,
    "num_epochs" : 1,
    "batch_size" : 16,
    "eval_freq" : 500, # how frequently evaluation to be run
    "eval_num_batches" : 20, # number of batches to run evaluation over at each go
}



