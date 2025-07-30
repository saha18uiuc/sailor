# SAILOR

Sailor is a framework that automates large-scale training over heterogeneous and geo-distributed resources. It is based on our SOSP'25 paper: 'Sailor: Automating Distributed Training over Dynamic,
Heterogeneous, and Geo-distributed Clusters'


It has 3 major components:
* A *Simulator* to estimate iteration time and memory footprint
* A *Planner* to decide on resource allocation and parallelization strategies on a pool of resources
* A *Training framework* supporting heterogeneous training, based on Megatron-DeepSpeed.

## SAILOR image creation

You can build the SAILOR image with:

* `git clone https://github.com/eth-easl/sailor.git`
* `cd sailor`
* `docker buildx build -t <image_name> .`

To build an image on the Alps cluster, follow the instructions in []()


If you use Sailor, please cite our paper:

```bibtex
@misc{sosp2025sailor,
      title={Sailor: Automating Distributed Training over Dynamic, Heterogeneous, and Geo-distributed Clusters},
      author={Foteini Strati and Zhendong Zhang and George Manos and Ixeia Sánchez Périz and Qinghao Hu and Tiancheng Chen and Berk Buzcu and Song Han and Pamela Delgado and Ana Klimovic},
      booktitle = {},
      year = {2025},
      isbn = {},
      address = {},
      pages = {},
      url = {},
      doi = {},
      publisher = {Association for Computing Machinery},
}
```