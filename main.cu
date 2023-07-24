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
    cudaEvent_t upload;
    cudaEvent_t done;
    cudaEvent_t check;
} Kernel;

typedef struct Delay {
    struct timespec whence;
    float estimate;
} Delay;

typedef struct State {
    int randFD;
    
    int numSMs;
    int threadsPerSM;
    int numThreads;
    size_t bufferSize;
    
    cudaStream_t transfer;
    Keypair *keys;

    cudaStream_t compute;
    Delay *delay;
} State;

void scheduleCompute(State s, Kernel kernel) {
    ssize_t seeded = read(s.randFD, s.keys, s.bufferSize);
    if (seeded != s.bufferSize) {
        fprintf(stderr, "read RNG error: %d instead of %d\n", seeded, s.bufferSize);
    }
    cudaMemcpyAsync(kernel.deviceMem, s.keys, s.bufferSize, cudaMemcpyHostToDevice, s.transfer);
    cudaEventRecord(kernel.upload, s.transfer);
    cudaStreamWaitEvent(s.compute, kernel.upload);
    search_kernel<<<s.numSMs, s.threadsPerSM, 0, s.compute>>>(kernel.deviceMem);
    cudaEventRecord(kernel.done, s.compute);
}

void checkResult(State s, Kernel kernel) {
    fprintf(stderr, "sleep\n");
    sleep(10.0);
    cudaStreamWaitEvent(s.transfer, kernel.done);
    cudaMemcpyAsync(s.keys, kernel.deviceMem, s.bufferSize, cudaMemcpyDeviceToHost, s.transfer);
    fprintf(stderr, "sync\n");
    cudaStreamSynchronize(s.transfer);
    cudaEventElapsedTime(&s.delay->estimate, kernel.upload, kernel.done);
    fprintf(stderr, "computed %d hashes in %fms\n", HASHES_PER_KERNEL * s.numThreads, s.delay->estimate);
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
    State s;
    Delay delay;
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
    delay.estimate = 11000.0;
    clock_gettime(CLOCK_MONOTONIC, &delay.whence);

    cudaStreamCreate(&s.transfer);
    cudaStreamCreate(&s.compute);
    cudaEventCreate(&primary.upload);
    cudaEventCreate(&primary.done);
    cudaEventCreate(&primary.check);
    cudaEventCreate(&secondary.upload);
    cudaEventCreate(&secondary.done);
    cudaEventCreate(&secondary.check);
    cudaMalloc((void**)&primary.deviceMem, s.bufferSize);
    cudaMalloc((void**)&secondary.deviceMem, s.bufferSize);

    scheduleCompute(s, primary);
    fprintf(stderr, "synchronize first transfer\n");
    cudaEventSynchronize(primary.upload);
    scheduleCompute(s, secondary);
    while (true) {
        checkResult(s, primary);
        scheduleCompute(s, primary);
        checkResult(s, secondary);
        scheduleCompute(s, secondary);
    }
}
