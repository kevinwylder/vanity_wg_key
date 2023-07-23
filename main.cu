#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/types.h>

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

int main() {

    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, 0);
    int threadsPerSM = 128; //deviceProps.maxThreadsPerMultiProcessor;
    int numSMs = deviceProps.multiProcessorCount;
    const int numThreads = threadsPerSM * numSMs;
    printf(
        "using %d threads on %d processors (%d total)\n",
        threadsPerSM, numSMs, numThreads
    );

    int random_fd = open("/dev/random", O_RDONLY);
    if (random_fd == -1) {
        printf("Error opening /dev/random\n");
        return 1;
    }

    size_t buffer_size = sizeof(Keypair) * numThreads;
    Keypair keys[numThreads];

    ssize_t bytes_read = read(random_fd, &keys, buffer_size);
    if (bytes_read == -1 || bytes_read < buffer_size) {
        printf("failed to read from /dev/random\n");
        return 2;
    }

    Keypair *device_keys;
    cudaMalloc((void**)&device_keys, buffer_size);
    cudaMemcpy(device_keys, &keys, buffer_size, cudaMemcpyHostToDevice);

    search_kernel<<<numSMs, threadsPerSM>>>(device_keys);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaDeviceSynchronize(); // Wait for kernel execution to finish
    printf("computed %d hashes\n", HASHES_PER_KERNEL * numThreads);

    cudaMemcpy(&keys, device_keys, buffer_size, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < numThreads; i++) {
        Keypair *key = &keys[i];
        if (key->rounds == 0) {
            continue;
        }
        printf("holy shit found it: %d %d\n", i, key->rounds);
        printf("A:\n");
        print_key(key->a, CURVE25519_KEY_SIZE);
        printf("B:\n");
        print_key(key->b, CURVE25519_KEY_SIZE);
        return;
    }
}
