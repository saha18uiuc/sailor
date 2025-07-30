## Reproducing results

### Simulator validation

### Figure 5a

This experiment validates SAILOR's and the other baselines' simulators on a cluster of GH200 GPUs.
The experiment tests memory estimation

```bash
bash ae_scripts/validation/run_gh200_mem.sh
python ae_scripts/validation/plot_box.py ae_results/validation/fig5a/ mem
```

The results and box plot are under /root/sailor/ae_results/validation/fig5a

### Figure 5b

Same as before, but now the experiment tests iteration time estimation

```bash
bash ae_scripts/validation/run_gh200_time.sh
python  ae_scripts/validation/plot_box.py ae_results/validation/fig5b/ time
```

The results and box plot are under /root/sailor/ae_results/validation/fig5b


### Figure 6

This experiment tests iteration time estimation on a cluster of heterogeneous gpus

```bash
bash ae_scripts/validation/run_het_time.sh
python  ae_scripts/validation/plot_box.py ae_results/validation/fig6/ mem
```

The results and box plot are under /root/sailor/ae_results/validation/fig6

### Planner

The following results use the Sailor simulator to evaluate Sailor's and the rest baselines' planners under different settings.

### Figure 7

Comparison between Sailor and the rest baselines on a cluster of A100 GPUs

```bash

bash ae_scripts/planner/run_homogeneous.sh
python ae_scripts/planner/plot_bars.py ae_results/planner/fig7/ OPT-350 homogeneous ae_results/planner/fig7/fig7.png

```

### Figure 8

Heterogeneous setup, with A100 + V100 GPUs, for the OPT-350M model

1. Figure 8a

```bash

bash ae_scripts/planner/run_het_opt.sh
python ae_scripts/planner/plot_bars.py ae_results/planner/fig8a/ OPT-350 heterogeneous ae_results/planner/fig8a/fig8a.png


```

2. Figure 8b

```bash


bash ae_scripts/planner/run_het_opt_imbalanced.sh
python ae_scripts/planner/plot_bars.py ae_results/planner/fig8b/ OPT-350 heterogeneous ae_results/planner/fig8b/fig8b.png


```

### Figure 10

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
