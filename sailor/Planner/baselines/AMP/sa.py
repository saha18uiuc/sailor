# pylint: disable-all
# flake8: noqa

# Script adapted from https://github.com/DachengLi1/AMP/blob/main/src/sa.py
from collections import defaultdict
from sailor.Planner.baselines.AMP.amp_utils import factor


def amp_no_placement_strategy(M, N, gbs, known):
    '''
        Function to obtain the next candidate for AMP without placement
        M: int, number of GPUs per node
        N: int, number of nodes
        gbs: int, global batch size
        known: dict, a dictionary of already obtained known candidates
    '''
    if known is None:
        known = defaultdict(list)
        ele_count = 0
        for h in factor(M):  # factor(M): # mp
            remain = M*N // h
            for w in factor(remain):  # DxP
                # print(gbs,w,factor(remain))
                if gbs >= w:
                    assert gbs % w == 0
                    for mbs in factor(gbs // w):  # all possible mbs for given gbs, DP
                        ele_count += 1
                        known[mbs].append((h, w))
        print(f"total possible amp candidates without placement: {ele_count}")
    if len(known.keys()) == 0:
        return None
    mbs = list(known.keys())[0]
    (h, w) = known[mbs].pop(0)
    if len(known[mbs]) == 0:
        known.pop(mbs, None)

    return h, w, mbs, known
