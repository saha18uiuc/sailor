/*%****************************************************************************
mpicxx test_nccl.cc -o test_nccl -lnccl -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda
%*****************************************************************************/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <unistd.h>

#include <nccl.h>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

#define CUDACHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        cudaError_t e = cmd;                                   \
        if (e != cudaSuccess)                                  \
        {                                                      \
            printf("Failed: Cuda error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#define NCCLCHECK(cmd)                                                                            \
    do                                                                                            \
    {                                                                                             \
        ncclResult_t r = cmd;                                                                     \
        if (r != ncclSuccess)                                                                     \
        {                                                                                         \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#define MPICHECK(cmd)                                                                                                  \
    do {                                                                                                               \
        int e = cmd;                                                                                                   \
        if (e != MPI_SUCCESS) {                                                                                        \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                                           \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

void test_func(int size, int rank, int world_size, ncclComm_t comm)
{
    void *buff;
    CUDACHECK(cudaMalloc(&buff, size));
    CUDACHECK(cudaMemset(buff, 0, size));
    int tag = 1000;
    MPI_Status status;

    double one_gb = 1e9 * 1.0;
    double size_gb = size / one_gb;

    int total_iters = 100;
    int warmup_iters = 3;
    int half_world = world_size / 2;

    CUDACHECK(cudaDeviceSynchronize());
    auto start = high_resolution_clock::now();

    // printf("Rank %d, Mem allocated, start test with size %d!\n", rank, size);
    for (int i = 0; i < total_iters; i++)
    {
        // auto start = high_resolution_clock::now();

        if (i == warmup_iters)
        {
	    MPI_Barrier(MPI_COMM_WORLD);
            CUDACHECK(cudaDeviceSynchronize());
	    // printf("Warmup done! start clock!\n");
	    start = high_resolution_clock::now();
        }

	/*int peer = 1 - rank;
	NCCLCHECK(ncclGroupStart());
	NCCLCHECK(ncclSend(buff, size, ncclUint8, peer, comm, 0));
        NCCLCHECK(ncclRecv(buff, size, ncclUint8, peer, comm, 0));	
	NCCLCHECK(ncclGroupEnd());*/

        if (rank < half_world)
        {
            NCCLCHECK(ncclSend(buff, size, ncclUint8, rank + half_world, comm, 0));
        }
        else
        {
            NCCLCHECK(ncclRecv(buff, size, ncclUint8, rank - half_world, comm, 0));
        }
        CUDACHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    CUDACHECK(cudaDeviceSynchronize());      
    auto stop = high_resolution_clock::now();

    double total_size_gb = (total_iters - warmup_iters) * half_world * size_gb;
    auto duration = duration_cast<nanoseconds>(stop - start);
    double duration_sec = duration.count() / (1e9 * 1.0);
    double bandwidth = total_size_gb / duration_sec;

    if (rank==0)
    	printf("%d,%f\n", size, bandwidth);
}

int main(int argc, char *argv[])
{


    int world_size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int half_world = world_size / 2;

    // nccl initialization
    ncclUniqueId id;
    ncclComm_t comm;
    if (rank == 0)
        ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    CUDACHECK(cudaSetDevice(rank % half_world));
    NCCLCHECK(ncclCommInitRank(&comm, world_size, id, rank));

    //printf("Rank %d joined, World size is %d!\n", rank, world_size);

    int n = stoi(argv[1]);
    double size = pow(2.0,n);
    test_func(size, rank, world_size, comm);
    usleep(100000);

    NCCLCHECK(ncclCommFinalize(comm));
    MPI_Finalize();

} /*main*/
