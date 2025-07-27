import torch
from deepspeed.pipe import PipelineModule

# TODO: not tested


def add_layers_recursively(module, layers):
    if isinstance(module, torch.nn.Sequential):
        module_list = [*module]
        for m in module_list:
            layers = add_layers_recursively(m, layers)
    else:
        layers.append(module)
    return layers

# VGG model
# TODO: this requires split based on parameters


class VGGPipeModelSpec(PipelineModule):
    def __init__(self, arch, **kwargs):
        specs = [
            *arch.features,
            arch.avgpool,
            lambda x: torch.flatten(x, 1),
            *arch.classifier,
        ]

        super().__init__(layers=specs, loss_fn=torch.nn.CrossEntropyLoss(), **kwargs)

# ViT model


class ViTPipeModelSpec(PipelineModule):
    def __init__(self, arch, **kwargs):
        specs = [arch.conv_proj, arch.custom_dropout]
        specs = add_layers_recursively(arch.encoder_layers, specs)
        specs.append(arch.custom_encoder_last)
        specs += arch.heads

        super().__init__(layers=specs, loss_fn=torch.nn.CrossEntropyLoss(), **kwargs)

# ConvNext model


class ConvNextPipeModelSpec(PipelineModule):
    def __init__(self, arch, **kwargs):
        specs = []
        specs = add_layers_recursively(arch.features, specs)
        specs.append(arch.avgpool)
        specs += arch.classifier

        super().__init__(layers=specs, loss_fn=torch.nn.CrossEntropyLoss(), **kwargs)
