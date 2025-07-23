import os

for i in range(1, 17):
    os.system(f"python3.9 ../../../profiling/repeat_prof.py profile_{i}_test.json 24 Varuna")
