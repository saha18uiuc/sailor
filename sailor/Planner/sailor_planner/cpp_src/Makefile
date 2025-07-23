CC=g++
CFLAGS=-O3
ARCH := $(shell uname -p)

planner.o: planner.cpp
	$(CC) $(CFLAGS) -fPIC -I/usr/include/python3.10 -c planner.cpp -ljsoncpp -o planner.o

training.o: training.cpp
	$(CC) $(CFLAGS) -fPIC -c training.cpp -ljsoncpp -o training.o

read_json.o: utils/read_json.cpp
	$(CC) $(CFLAGS) -fPIC -c utils/read_json.cpp -ljsoncpp -o read_json.o

network_utils.o: utils/network_utils.cpp
	$(CC) $(CFLAGS) -fPIC -c utils/network_utils.cpp -o network_utils.o

planner_utils.o: planner_utils.cpp
	$(CC) $(CFLAGS) -fPIC -c planner_utils.cpp -ljsoncpp -o planner_utils.o

libplanner.so: planner.o read_json.o training.o planner_utils.o network_utils.o
	$(CC) $(CFLAGS) -fPIC -shared -I/usr/include/python3.10  $(python3 -m pybind11 --includes) read_json.o training.o planner_utils.o network_utils.o planner.o -ljsoncpp -o libplanner.cpython-310-$(ARCH)-linux-gnu.so

all:
	make libplanner.so

clean:
	rm -rf *.o *.so
