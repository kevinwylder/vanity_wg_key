#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/types.h>
#include <time.h>

#include "curve25519.cu"
#include "base64.c"

#define HASHES_PER_KERNEL 10000

__host__ __device__ bool match(uint8_t *key) {
    return (
        key[0] == 0x93 && 
        key[1] == 0x0c && 
        key[2] == 0xa5 && 
        key[3] == 0x75 && 
        key[4] == 0xea && 
        key[5] >= 0xc0
   );
}

typedef struct Keypair {
    uint8_t a[CURVE25519_KEY_SIZE];
    uint8_t b[CURVE25519_KEY_SIZE];
    bool privkey_in_b;
    uint32_t rounds;
} Search;

__device__ void do_search(Keypair *key) {
    uint8_t *a = &key->a[0];
    uint8_t *b = &key->b[0];
    while (--key->rounds > 0) {
        curve25519_generate_public(a, b);
        key->privkey_in_b = true;
        if (match(a)) {
            return;
        }
        curve25519_generate_public(b, a);
        key->privkey_in_b = false;
        if (match(b)) {
            return;
        }
    }
}

__global__ void search_kernel(Keypair *keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Keypair *key = &keys[tid];
    key->rounds = HASHES_PER_KERNEL / 2;
    do_search(key);
}

typedef struct Kernel {
    Keypair *deviceMem;
    cudaEvent_t ready;
    cudaEvent_t start;
    cudaEvent_t finish;
} Kernel;

typedef struct State {
    int randFD;
    
    int numSMs;
    int threadsPerSM;
    int numThreads;
    size_t bufferSize;
    
    cudaStream_t transfer;
    Keypair *keys;

    cudaStream_t compute;
    float *delay;
} State;

void scheduleCompute(State s, Kernel kernel) {
    ssize_t seeded = read(s.randFD, s.keys, s.bufferSize);
    if (seeded != s.bufferSize) {
        fprintf(stderr, "read RNG error: %d instead of %d\n", seeded, s.bufferSize);
    }
    cudaMemcpyAsync(kernel.deviceMem, s.keys, s.bufferSize, cudaMemcpyHostToDevice, s.transfer);
    cudaEventRecord(kernel.ready, s.transfer);
    cudaStreamWaitEvent(s.compute, kernel.ready);
    cudaEventRecord(kernel.start, s.compute);
    search_kernel<<<s.numSMs, s.threadsPerSM, 0, s.compute>>>(kernel.deviceMem);
    cudaEventRecord(kernel.finish, s.compute);
}

void checkResult(State s, Kernel kernel) {
    float delay = *s.delay * 0.00099;
    sleep(delay);
    cudaStreamWaitEvent(s.transfer, kernel.finish);
    cudaMemcpyAsync(s.keys, kernel.deviceMem, s.bufferSize, cudaMemcpyDeviceToHost, s.transfer);
    cudaStreamSynchronize(s.transfer);
    cudaEventElapsedTime(s.delay, kernel.start, kernel.finish);
    fprintf(stderr, "computed %d hashes in %fms\n", HASHES_PER_KERNEL * s.numThreads, *s.delay);
    for (size_t i = 0; i < s.numThreads; i++) {
        Keypair *key = &s.keys[i];
        if (key->rounds == 0) {
            continue;
        }
        fprintf(stderr, "Found a match!!! %d %d\n", i, key->rounds);
        print_key(key->a, CURVE25519_KEY_SIZE);
        print_key(key->b, CURVE25519_KEY_SIZE);
        exit(0);
    }
}

int main() {
    // initialize state 
    float delay = 11000.0;
    State s;
    Kernel primary;
    Kernel secondary;

    s.randFD = open("/dev/urandom", O_RDONLY);

    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, 0);
    s.threadsPerSM = 128; //deviceProps.maxThreadsPerMultiProcessor;
    s.numSMs = deviceProps.multiProcessorCount;
    s.numThreads = s.threadsPerSM * s.numSMs;
    s.bufferSize = s.numThreads * sizeof(Keypair);
    s.keys = (Keypair *) malloc(s.bufferSize);

    s.delay = &delay;

    cudaStreamCreate(&s.transfer);
    cudaStreamCreate(&s.compute);
    cudaEventCreate(&primary.ready);
    cudaEventCreate(&primary.start);
    cudaEventCreate(&primary.finish);
    cudaEventCreate(&secondary.ready);
    cudaEventCreate(&secondary.start);
    cudaEventCreate(&secondary.finish);
    cudaMalloc((void**)&primary.deviceMem, s.bufferSize);
    cudaMalloc((void**)&secondary.deviceMem, s.bufferSize);

    scheduleCompute(s, primary);
    cudaEventSynchronize(primary.ready);
    scheduleCompute(s, secondary);
    while (true) {
        checkResult(s, primary);
        scheduleCompute(s, primary);
        checkResult(s, secondary);
        scheduleCompute(s, secondary);
    }
}
