**Validation in Clariden/MIT Clusters**

**Profile Iteration Time and Memory**

In sailor/Planner/baselines/Aceso, run
```
bash scripts_clariden/run_clariden.sh
```

The results are under aceso_validation_clariden/runtime/OPT-350. 
 
Note: You can download the results and logout from the Alps cluster, all remaining experiments will run on a Sailor container in any machine.



**Generate Aceso Simulation Results**
```
bash scripts_clariden/aceso_opt_simulate.sh
bash scripts_mit/aceso_opt_simulate.sh
```

**Compare with Real Time and Memory**
```
python scripts_clariden/compare_aceso.py 
python scripts_mit/compare_aceso.py 
```
