NCCL_PATH = /usr/local/nccl2
CUDA_PATH = /usr/local/cuda

test_nccl: test_nccl.cc
	mpicxx test_nccl.cc -o test_nccl -I$(NCCL_PATH)/include -L$(NCCL_PATH)/lib  -lnccl -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcudart -lcuda

clean:
	rm test_nccl

