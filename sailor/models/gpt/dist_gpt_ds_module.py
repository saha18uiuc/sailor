from deepspeed.pipe import LayerSpec, PipelineModule
from .gpt_modules import GPTTransformerLayer, GPTEmbedding, GPTLMHead
from torch import nn


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


class GPTPipeModelSpec(PipelineModule):
    def __init__(self, vocab_size, embedding_dim, seq_length, num_heads, num_layers, **kwargs):
        specs = [
            LayerSpec(GPTEmbedding, vocab_size, embedding_dim, seq_length)
        ]

        for _ in range(num_layers):
            specs.append(LayerSpec(
                GPTTransformerLayer, embedding_dim, num_heads, embedding_dim * 4))

        specs.append(LayerSpec(GPTLMHead, embedding_dim, vocab_size))
        super().__init__(layers=specs, loss_fn=LMLoss(False), **kwargs)


class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length, num_heads, num_layers):
        super().__init__()
        self.embedding = GPTEmbedding(vocab_size, embedding_dim, seq_length)
        self.layers = nn.Sequential(*[
            GPTTransformerLayer(embedding_dim, num_heads, embedding_dim * 4)
            for _ in range(num_layers)
        ])
        self.head = GPTLMHead(embedding_dim, vocab_size)

    def forward(self, input_ids, targets):
        x = self.embedding(input_ids)
        x = self.layers(x)
        return self.head(x, targets)
