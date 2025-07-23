## Information about data for multizone
### Cost data transfer:
`cost_data_transfer.csv`
Grabbed from `https://github.com/eth-easl/sailor_opt/blob/main/data/network_cost.csv`
Simplified as follows:
- node in the same zone: taken as us-west1-b <-> us-west-1b or -1c (assume it is the same)
- nodes in different zone, same region: taken as us-west1-a <-> us-central1-a
- nodes in differnet zone and different regions: taken as us-west1-a <-> europe-west1-a
#TODO: not included in the objective, we need to compute the total GBs sent and add the cost

### Cost for bandwidth in VMs:
Grabbed from `https://github.com/eth-easl/sailor_opt/tree/main/cloud_profiling/networking/nccl/nccl_tests`
Simplified as follows:
- `p2p_same_region` and `p2p_same_region_std`: nodes in different zone, same region. 
- `p2p_diff_region` and `p2p_diff_region_std`: nodes in different zones and different regions

We did not profile the measures in Sailor and we construct the files as follows:
- `bw_profiling_V100_sailor_Gbps_MZ.csv`
    - `p2p_same_region`, different zone, same region0: from `cloud_profiling/networking/nccl/nccl_tests/V100/results/send_recv/1_threads/1R-us-central1-a-us-central1-b.csv` column `algbw_oop` (assume GB/sec)
        * NOTE: the bndwidth for `us-central1-a-us-west1-b` were much lower for some reason? so I avoided taking them because it was lower than crossing regions. 
    - `p2p_diff_region`, different region: `cloud_profiling/networking/nccl/nccl_tests/V100/results/send_recv/1_threads/2R-us-central1-a-us-west1-b.csv`, column `algbw_oop` (assume GB/sec)
`bw_profiling_A100-40_sailor_Gbps_MZ.csv`
    - `p2p_same_region`, different zone, same region: `cloud_profiling/networking/nccl/nccl_tests/A100/a100_same_region/threads1.csv`, column `BW1(GB/sec)`
    - `p2p_diff_region`, different region: `cloud_profiling/networking/nccl/nccl_tests/A100/a100_us_eu/threads1.csv`, column `BW1(GB/sec)`