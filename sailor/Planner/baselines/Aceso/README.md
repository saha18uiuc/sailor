**Validation in Clariden Cluster**

**1. Profile Iteration Time and Memory**

In sailor/Planner/baselines/Aceso, run
```
bash scripts_clariden/run_clariden.sh
```

The results are under aceso_validation_clariden/runtime/OPT-350.

**2. Download the results and logout from the Alps cluster, all remaining experiments will run on a Sailor container in any machine.**


**3. Generate Aceso Simulation Results**
```
bash scripts_clariden/aceso_opt_simulate.sh
```

**4. Compare with Real Time and Memory**
```
python scripts_clariden/compare_aceso.py
```
