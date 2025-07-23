from typing import Tuple
from sailor.Planner.baselines.FlashFlex.src.cost_modeling import *
from sailor.Planner.baselines.FlashFlex.src.globals import *


def throughput(all_pipelines, configs):
    """given pipelines, calculate the throughput"""
    if all_pipelines is None:
        return 0

    model_time_cost = TimeCost(all_pipelines, configs)

    token_throughput = model_time_cost.token_throughput()

    return token_throughput
