from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=False)
class LLM:
    name: str
    hidden_dim: int
    seq_length: int
    n_heads: int
    transformer_layers: int
    number_of_parameters: int
    vocab_size: int
    ffn_dim: int


@dataclass(frozen=True)
class Vision:
    pass


@dataclass(frozen=False)
class Training:
    model: object
    optimizer: str
    global_batch_size: int
    bytes_per_parameter: int
    ckpt_overhead: float
    pricing_strategy: int
    objective: str
    max_cost: float
    min_throughput_percentage: float
    activation_recomputation: bool
    gpu_names: List[str]
    embedding_mem: bool
    transformer_mem: bool
    num_layers: int
    head_mem: bool
    heterogeneity: Optional[dict] = None


# Define model
# NOTE: for gpt2 we set ffn_dim = n_embd*4
# from this file https://huggingface.co/openai-community/gpt2-medium/blob/main/config.json
gpt2 = LLM(name='gpt2', hidden_dim=768, seq_length=1024, transformer_layers=12, n_heads=16,
           ffn_dim=768*4, number_of_parameters=131.170*10**6, vocab_size=52256)
gpt2_medium = LLM(name='gpt2-medium', hidden_dim=1024, seq_length=1024, n_heads=20,
                  transformer_layers=24, ffn_dim=1024*4, number_of_parameters=345*10**6, vocab_size=52256)
gpt2_large = LLM(name='gpt2-large', hidden_dim=1280, seq_length=1024, n_heads=20,
                 transformer_layers=36, ffn_dim=1280*4, number_of_parameters=774*10**6, vocab_size=52256)
gpt2_xl = LLM(name='gpt2-xl', hidden_dim=1600, seq_length=1024, transformer_layers=48, n_heads=25,
              ffn_dim=1600*4, number_of_parameters=15.58*10**8, vocab_size=52256)
gpt2_xl_20b = LLM(name='gpt2-xl-20b', hidden_dim=1600*4, seq_length=1024*4, n_heads=25,
                  transformer_layers=48, ffn_dim=1600*16, number_of_parameters=20*10**9, vocab_size=52256)
gpt_neo27b = LLM(name='GPT-Neo-2.7', hidden_dim=2560, seq_length=2048, n_heads=20,
                 transformer_layers=32, ffn_dim=2560*16, number_of_parameters=2.7*10**9, vocab_size=50257)

# NOTE: For Mistral, the ffn_dim is obtained by multiplying `hidden_dim*8` as the model has 8 heads
mistral7b = LLM(name='mistral7b', hidden_dim=4096,  seq_length=2048, n_heads=20,
                transformer_layers=48, ffn_dim=4096*8, number_of_parameters=7*10**9, vocab_size=32000)

# Ref OPT: https://arxiv.org/pdf/2205.01068.pdf (Table 1), https://huggingface.co/facebook/opt-6.7b/blob/main/config.json
opt125m = LLM(name='opt125m', hidden_dim=768, seq_length=2048, n_heads=12,
              transformer_layers=12, ffn_dim=3072, number_of_parameters=125*10**6, vocab_size=28998)
opt350m = LLM(name='OPT-350', hidden_dim=1024, seq_length=2048, n_heads=16,
              transformer_layers=24, ffn_dim=4096, number_of_parameters=350*10**6, vocab_size=28998)
opt_1_3b = LLM(name='OPT-1.3B', hidden_dim=2048, seq_length=2048, n_heads=24,
               transformer_layers=24, ffn_dim=8192, number_of_parameters=1.3*10**9, vocab_size=28998)
opt_2_7b = LLM(name='OPT-2.7B', hidden_dim=2560, seq_length=2048, n_heads=32,
               transformer_layers=32, ffn_dim=10240, number_of_parameters=2.7*10**9, vocab_size=28998)
opt_6_7b = LLM(name='OPT-6.7B', hidden_dim=4096, seq_length=2048, n_heads=40,
               transformer_layers=32, ffn_dim=16384, number_of_parameters=6.7*10**9, vocab_size=28998)
opt13b = LLM(name='OPT13B', hidden_dim=5120,  seq_length=2048, n_heads=40,
             transformer_layers=40, ffn_dim=20480, number_of_parameters=13*10**9, vocab_size=28998)
opt30b = LLM(name='OPT30B', hidden_dim=7168,  seq_length=2048, n_heads=56,
             transformer_layers=48, ffn_dim=28672, number_of_parameters=30*10**9, vocab_size=28998)
opt66b = LLM(name='OPT66B', hidden_dim=9216,  seq_length=2048, n_heads=72,
             transformer_layers=64, ffn_dim=36864, number_of_parameters=66*10**9, vocab_size=28998)
opt175b = LLM(name='OPT175B', hidden_dim=12288,  seq_length=2048, n_heads=96,
              transformer_layers=96, ffn_dim=49152, number_of_parameters=175*10**9, vocab_size=28998)

llama3_8b = LLM(name='LLAMA-3-8', hidden_dim=4096,  seq_length=2048, n_heads=32,
                transformer_layers=32, ffn_dim=16384, number_of_parameters=8*10**9, vocab_size=128256)

model_mapping = {
    gpt2.name: gpt2,
    gpt2_xl.name: gpt2_xl,
    gpt2_medium.name: gpt2_medium,
    gpt2_large.name: gpt2_large,
    gpt2_xl_20b.name: gpt2_xl_20b,
    gpt_neo27b.name: gpt_neo27b,
    mistral7b.name: mistral7b,
    opt125m.name: opt125m,
    opt350m.name: opt350m,
    opt_1_3b.name: opt_1_3b,
    opt_2_7b.name: opt_2_7b,
    opt_6_7b.name: opt_6_7b,
    opt13b.name: opt13b,
    opt30b.name: opt30b,
    opt66b.name: opt66b,
    opt175b.name: opt175b,
    llama3_8b.name: llama3_8b
}
