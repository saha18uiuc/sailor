# Adapted https://github.com/eth-easl/sailor_opt/blob/extend_simulator_to_multi_region_dp_pp/sailor_prof/llm_ds/convert_to_ds.py
# and sailor.models.gpt2.dist_gpt_ds_module

from deepspeed.pipe import LayerSpec, PipelineModule

from torch import nn
from sailor.models.opt.opt_modules import GPTEmbeddings, GPTBlock, GPTLMHead


class LMCrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, is_opt, *args, **kwargs):
        self.is_opt = is_opt
        super().__init__(*args, **kwargs, ignore_index=-100)

    # pylint: disable=redefined-builtin
    def forward(self, input, target):
        if not self.is_opt:
            shift_logits = input[..., :-1, :].contiguous()
            shift_labels = target[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            return super().forward(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length)
            )
        shift_logits = input[..., :].contiguous()
        shift_labels = target[..., :].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss


class LMLoss(nn.Module):

    def __init__(self, is_opt):
        super().__init__()
        self.crit_ce = LMCrossEntropyLoss(is_opt)

    def forward(self, lm_logits, labels):
        loss = self.crit_ce(lm_logits, labels)
        return loss


class OptPipeModelSpec(PipelineModule):
    def __init__(self, config, res, **kwargs):
        specs = []
        emb = GPTEmbeddings(config, res=res, ind=0)
        specs.append(emb)

        for i in range(1, config.num_hidden_layers + 1):
            specs.append(LayerSpec(GPTBlock, config, res, ind=i))
        ind = config.num_hidden_layers + 1
        head = GPTLMHead(config, res=res, ind=ind)
        specs.append(head)
        super().__init__(layers=specs, loss_fn=LMLoss(True), **kwargs)
