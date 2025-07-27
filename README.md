# SAILOR

Sailor is a framework that automates large-scale training over heterogeneous and geo-distributed resources.
It has 3 major components:
* A *Simulator* to estimate iteration time and memory footprint
* A *Planner* to decide on resource allocation and parallelization strategies on a pool of resources
* A *Training framework* supporting heterogeneous training

## SAILOR image creation

You can build the SAILOR image with:

* `git clone https://github.com/eth-easl/sailor.git`
* `cd sailor`
* `docker buildx build -t <image_name> .`

To build an image on the Alps cluster, follow the instructions in []()