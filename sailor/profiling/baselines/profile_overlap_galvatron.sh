#!/bin/bash

# with 2
python profile_overlap.py --local-rank 0 --world-size 2 --master-addr 127.0.0.1 --master-port 1234  --coef-file /root/sailor/sailor/Planner/baselines/Galvatron/overlap_coe_dicts.json --gpu A100 &
python profile_overlap.py --local-rank 1 --world-size 2 --master-addr 127.0.0.1 --master-port 1234  --coef-file /root/sailor/sailor/Planner/baselines/Galvatron/overlap_coe_dicts.json --gpu A100 &

# with 4
# python profile_overlap.py --local-rank 0 --world-size 4 --master-addr 127.0.0.1 --master-port 1234  --coef-file /root/sailor/sailor/Planner/baselines/Galvatron/overlap_coe_dicts.json --gpu A100 &
# python profile_overlap.py --local-rank 1 --world-size 4 --master-addr 127.0.0.1 --master-port 1234  --coef-file /root/sailor/sailor/Planner/baselines/Galvatron/overlap_coe_dicts.json --gpu A100 &
# python profile_overlap.py --local-rank 2 --world-size 4 --master-addr 127.0.0.1 --master-port 1234  --coef-file /root/sailor/sailor/Planner/baselines/Galvatron/overlap_coe_dicts.json --gpu A100 &
# python profile_overlap.py --local-rank 3 --world-size 4 --master-addr 127.0.0.1 --master-port 1234  --coef-file /root/sailor/sailor/Planner/baselines/Galvatron/overlap_coe_dicts.json --gpu A100 &
