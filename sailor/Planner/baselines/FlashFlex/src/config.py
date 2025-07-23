from sailor.Planner.baselines.FlashFlex.src.initialize import initialize
from pymetis import Options
from dataclasses import dataclass


class Config:
    def __init__(self, training_config, niter, npipeline, kway, inter_bw, micro_bsz):

        # graph partition config
        self.options = Options(contig=True)
        self.npipeline = npipeline
        self.param = [2, 0.2]  # n, p for binomial
        self.K = initialize(self.param, (1, npipeline))
        self.specs = None
        self.niter = niter

        self.kway = kway
        self.inter_bw = inter_bw

        # utils
        self.device_machine_map = None

        # model config

        self.global_bsz = training_config["global_batch_size"]
        accum_iter = 1
        self.GLB_B = self.global_bsz * accum_iter
        # GLB_B = 5000
        self.GLB_MB = micro_bsz * accum_iter

        assert self.GLB_B % self.GLB_MB  == 0
        self.N_MB = self.GLB_B // self.GLB_MB

        print(f"CONFIG-INIT, GLB_B is {self.GLB_B}, GLB_MB is {self.GLB_MB}, N_MB is {self.N_MB}")

        assert self.GLB_MB >= npipeline, "Too many pipelines"
        self.B = self.GLB_B // self.npipeline
        self.MB = self.GLB_MB // self.npipeline

        self.S = training_config["sequence_length"]
        self.H = training_config["hidden_size"]
        self.L = training_config["num_layers"]  # original code does not include Embedding + head
        self.N_attn_heads = training_config["heads"]

        self.V = training_config["vocab_size"]
        self.B_type = 4 # float32
        self.T = self.GLB_B * self.S