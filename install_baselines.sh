#!/bin/bash
cd sailor/Planner/baselines/Galvatron/csrc
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) dp_core.cpp -o galvatron_dp_core.cpython-310-$(uname -m)-linux-gnu.so
cd $HOME/sailor