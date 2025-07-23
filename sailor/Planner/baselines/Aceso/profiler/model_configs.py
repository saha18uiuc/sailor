# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

## "algo" stands for tensor parallel partition algorithm
model_prof_configs = {
    "GPT-Neo-2.7": {
        "mbs": [1, 2, 4, 8],
        "algo": [0, 1], 
        # model_size: (num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size)
        # "configs": (1, 2048, 2560, 2560*4, 32, 2560//32, 128256)
    },
    "OPT-30": { 
        "mbs": [1, 2, 4, 8],
        "algo": [0, 1],
        # model_size: (num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size)
        # "configs": (1, 2048, 7168, 7168*4, 56, 7168//56, 50272)
    }, 
    "OPT-350": { 
        "mbs": [1, 2, 4, 8],
        "algo": [0, 1],
        # model_size: (num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size)
        # "configs": (1, 2048, 1024, 1024*4, 16, 1024//16, 50272)
    }
}