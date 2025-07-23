# pylint: disable-all

# Script adapted from https://github.com/DachengLi1/AMP/blob/main/src/pipe.py
import copy
import time

import torch
import numpy as np
from math import floor


def partition_uniform(num_items, num_parts):
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    chunksize = floor(num_items / num_parts)
    for p in range(num_parts):
        parts[p] = min(chunksize * p, num_items)
    parts[num_parts] = num_items
    return parts


# Dynamic programming algorithm that supports 3D parallelism, and any split of layers
def pipe_ast(L, cost_e, cost_c, k, B):
    time_dp_s = time.time()
    possible = [0]

    for i in range(1, L+1):
        ptr = 0
        while ptr + i <= L:
            possible.append(sum(cost_e[ptr:ptr+i]))
            ptr += 1
    # TODO: constraint to uniform partitionlayers
    possible = sorted(list(set(possible)))
    cutpoints = partition_uniform(L, k)

    trace = []
    for i in range(L):
        outer = []
        for j in range(k):
            inner = []
            for m in range(len(possible)):
                inner.append(([], np.infty))
            outer.append(inner)
        trace.append(outer)

    for i in range(L):
        for j in range(k):
            for m in range(len(possible)):
                if i+1 <= j:  # invalid
                    pass
                else:
                    if j == 0:  # base case: 0 cut
                        cur_sum = sum(cost_e[:i+1])
                        assert cur_sum in possible
                        trace[i][j][m] = (
                            [i+1], (B-1) * max(0, cur_sum - possible[m]))
                    else:
                        cost_best = np.infty
                        S_best = []
                        for cut in range(j-1, i):
                            cur_sum = sum(cost_e[cut+1:i+1])
                            assert cur_sum in possible
                            S, cost_ = trace[cut][j -
                                                  1][possible.index(max(cur_sum, possible[m]))]
                            cost_ += (B-1) * max(0, cur_sum - possible[m])
                            cost_ += cost_c[cut][j-1]
                            if cost_ < cost_best:
                                cost_best = cost_ - cost_c[cut][j-1]
                                S_ = copy.deepcopy(S)
                                S_.append(i-cut)
                                S_best = S_
                        trace[i][j][m] = (S_best, cost_best)

    time_dp_used = time.time() - time_dp_s

    # add each stage cost at the end
    S, cost = trace[L-1][k-1][0]
    cost += np.sum(cost_e)
    print(
        f"pipe_ast used {round(time_dp_used,2)} seconds with {L} layers and {k} stages.")
    return (S, cost)


# DeepSpeed uniform partition (currently used)
def pipe_ds(L, k):
    per_stage = L // k
    s = [per_stage] * (k-1)
    s.append(int(L)-sum(s))
    return s

    for i in range(1, k):
        p.append(p[i-1] + s[i])
    lens = torch.reshape(torch.sum(cost_e[:p[0]+1]), (-1, 1))

    for i in range(len(s)-1):
        lens = torch.cat([lens, torch.reshape(
            torch.sum(cost_e[p[i]+1:p[i+1]+1]), (-1, 1))])

    max_l = torch.max(lens)
    cost = (B-1) * max_l
    for i in range(k-1):
        cost += cost_c[p[i]][i]
    cost += torch.sum(cost_e)
    return s, cost


def pipe_gpt2(L, pp):
    each = L // pp
    remain = L - pp * each
    start = 2
    ret = [start + each]
    for i in range(pp-1):
        ret.append(each)
    for i in range(remain):
        ret[i] += 1
    ret[-1] += 4
    # print(f"-----------{ret}. {L}, {pp}")
    return ret, None


def pipe_uniform(L, pp):
    # print("using a uniform")
    each = L // pp
    remain = L - pp * each
    ret = [each]
    for i in range(pp-1):
        ret.append(each)
    for i in range(remain):
        ret[i] += 1
    # print(f"pipe uniform returns {ret}")
    # print(f"-----------{ret}. {L}, {pp}")
    return ret, None


def pipe_transgan(cost_e, pp):
    # buggy
    assert False
    each = np.sum(cost_e) // pp
    assignment = []
    cumulative_time = 0
    cumulative_length = 0
    print(cost_e)
    print(each)
    for i in range(len(cost_e)):
        cumulative_time += cost_e[i]
        cumulative_length += 1
        if cumulative_time >= each:
            assignment.append(cumulative_length)
            cumulative_time = 0
            cumulative_length = 0
        print(cumulative_time, cumulative_length)
    # remain =
    if cumulative_length != 0:
        assignment.append(cumulative_length)
    return assignment, None


def pipe_cost(L, cost_e, cost_c, k, B, partition):
    '''
    L: number of layers
    cost_e: execution cost of each layer (L,)
    cost_c: communication cost of each layer of each stage (L-1, k-1)
    k: number of stages
    B: number of microbatches
    partition: number of layers per stage (k,)

    Output: cost of forward, backward, communication and ga for this pipeline configuration
    '''
    s = partition
    p = [s[0]-1]

    for i in range(1, int(k.item())):
        p.append(p[i-1] + s[i])
    # sum execution of first stage
    lens = torch.reshape(torch.sum(cost_e[:p[0]+1]), (-1, 1))
    # print(f"calculating cost: {cost_e} {cost_c} {k} {B} {partition}")
    for i in range(len(s)-1):
        lens = torch.cat([lens, torch.reshape(
            torch.sum(cost_e[p[i]+1:p[i+1]+1]), (-1, 1))])

    max_l = torch.max(lens)
    cost = (B-1) * max_l
    for i in range(int(k.item())-1):
        cost += cost_c[p[i]][i]
    cost += torch.sum(cost_e)
    return cost
