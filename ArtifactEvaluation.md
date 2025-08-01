## Creating the image

We provide a Dockerfile with all dependencies needed for Sailor.
When using docker, create the image by:

```bash
git clone https://github.com/eth-easl/sailor.git
cd sailor && git checkout sosp25_ae
docker buildx build -t <image_name> .
```

## Minimal functional example

For a functional example, and step 1 of the following Reproducing results section, we will use the the [Alps Clariden](https://docs.cscs.ch/clusters/clariden/#logging-into-clariden) cluster.
After creating an account and being added to our project, follow the instructions in [create_image_alps.md](create_image_alps.md) to create an image in the Alps cluster, get a node, and run a simple example.

## Reproducing results

We provide scripts that automate experiments for key figures in the paper.
Please note that the Planner evaluation experiments do not depend on the Simulator validation experiments, so you can run them in any order.

## 1. Simulator validation

For this stage, an image need to have been built in Clariden, as described above. We will first run some training jobs in Clariden to measure iteration time and memory bandwidth, then we will process the results and update the testing configuration files, and then we will validate the Sailor and other simulators.

Notes:
* Since Aceso needs seperate runs, we omit it to save time
* We have done the necessary profiling of the model and cluster network

### Get results on the Clariden cluster

1. Login to Clariden and go to '/capstor/scratch/cscs/$USER/sailor' directory

2.  Run the script [ae_scripts/clariden_scripts/run.sh](ae_scripts/clariden_scripts/run.sh). This submits slurm jobs of the Sailor framework with different configurations. We are testing both iteration time and memory footprint. You can use 'squeue' to check the status of your jobs.

**Note: If you are running on a different project than a-infra02, please replace a-infra02 with your project ID in [ae_scripts/clariden_scripts/run.sh](ae_scripts/clariden_scripts/run.sh) and [ae_scripts/clariden_scripts/run_opt.slurm](ae_scripts/clariden_scripts/run_opt.slurm)**

### Process results
After all jobs have finished, we have to process the results. The results for both iteration time and memory configurations are under /capstor/scratch/cscs/$USER/sailor/clariden/OPT-350.

1. Copy the files to the host you want to evaluate the rest of the artifact (no GPU required). Clone the sailor repo and build the image (e.g. with docker, as mentioned above).

2. Start a container using the Sailor image

3. Once inside the container, do:

```bash
cd ae_scripts/clariden_scripts
bash process_results.sh <results_dir_to_OPT-350> ../../sailor/Planner/simulations/validation/clariden/OPT-350/
```

Replace 'results_dir_to_OPT-350' with your copied directory from step 1. This will overwrite the testing configurations under sailor/Planner/simulations/validation/clariden/OPT-350/, which will be used for validation in the following steps.

**Note: from now on, you should be inside the Sailor container**

### Validation - Figure 5a

This experiment validates SAILOR's and the other baselines' simulators on a cluster of GH200 GPUs.
The experiment tests memory estimation

```bash
bash ae_scripts/validation/run_gh200_mem.sh
python ae_scripts/validation/plot_box.py ae_results/validation/fig5a/ mem
```

The results and box plot are under /root/sailor/ae_results/validation/fig5a

### Validation - Figure 5b

Same as before, but now the experiment tests iteration time estimation

```bash
bash ae_scripts/validation/run_gh200_time.sh
python  ae_scripts/validation/plot_box.py ae_results/validation/fig5b/ time
```

The results and box plot are under /root/sailor/ae_results/validation/fig5b

## 2. Planner evaluation

The following results use the Sailor simulator to evaluate Sailor's and the rest baselines' planners under different settings. You need to be inside a Sailor container to run the following scripts

### Homogeneous setup - Figure 7

Comparison between Sailor and the rest baselines on a cluster of A100 GPUs

```bash
bash ae_scripts/planner/run_homogeneous.sh
python ae_scripts/planner/plot_bars.py ae_results/planner/fig7/ OPT-350 homogeneous ae_results/planner/fig7/fig7.png
```

### Heterogeneous setup - Figure 8b

Heterogeneous setup, with A100 + V100 GPUs, for the OPT-350M model


```bash
bash ae_scripts/planner/run_het.sh
python ae_scripts/planner/plot_bars.py ae_results/planner/fig8b/ OPT-350 heterogeneous-imbalanced ae_results/planner/fig8b/fig8b.png
```

### Geodistributed setup - Figure 10

Comparison between DTFM and Sailor on a geo-distributed setup (A100 GPUs across 5 zones)

```bash
bash ae_scripts/planner/run_geo.sh
python ae_scripts/planner/plot_bars.py ae_results/planner/fig10/ OPT-350 geo ae_results/planner/fig10/fig10.png
```

The results and plot are under /root/sailor/ae_results/validation/fig10.
Note: DTFM uses random values, so the result configurations might be different from the ones in Fig 10, and this is expected (you can try rerunning to check). We have run DTFM multiple times and show huge performance gap compared to SAILOR.

### Figure 11

Cost minimization with minimum throughput constraint

```bash
bash ae_scripts/planner/run_fig11.sh
python ae_scripts/planner/plot_mul_obj.py ae_results/planner/fig11/ OPT-350 dollars_per_iter ae_results/planner/fig11/fig11.png
```

### Figure 12

Throughput maximization with maximum budget

```bash
bash ae_scripts/planner/run_fig12.sh
python ae_scripts/planner/plot_mul_obj.py ae_results/planner/fig12/ OPT-350 total_throughput ae_results/planner/fig12/fig12.png
```
