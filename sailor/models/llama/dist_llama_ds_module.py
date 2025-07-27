from deepspeed.pipe import LayerSpec, PipelineModule
from torch import nn

from sailor.models.llama.llama_modules import LlamaDecoderLayerTupleIO, LlamaEnter, LlamaExit


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


# llama does not tie weights
class LlamaPipeModelSpec(PipelineModule):

    def __init__(self, config):
        specs = []
        specs.append(LlamaEnter(config, res={}, ind=0))

        for i in range(1, config.num_hidden_layers+1):
            specs.append(LayerSpec(LlamaDecoderLayerTupleIO, config, ind=i))

        ind = config.num_hidden_layers + 1
        specs.append(LlamaExit(config, res={}, ind=ind))
        super().__init__(layers=specs, loss_fn=LMLoss(True),  num_stages=1, partition_method='uniform')
