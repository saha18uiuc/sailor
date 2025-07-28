## Reproducing results


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
