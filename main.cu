#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/types.h>

#include "curve25519.cu"
#include "base64.c"

typedef struct Keypair {
    uint8_t a[CURVE25519_KEY_SIZE];
    uint8_t b[CURVE25519_KEY_SIZE];
    bool odd;
    bool found;
} Search;

__device__ void search(Keypair *key) {
    key->odd = true;
    curve25519_generate_public((uint8_t *) &key->a, (const uint8_t *) &key->b);
    key->odd = false;
    curve25519_generate_public((uint8_t *) &key->b, (const uint8_t *) &key->a);
}

__global__ void search_kernel(Keypair *keypair) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Keypair *key = &keypair[tid];
    search(key);
}

int main() {
    int random_fd = open("/dev/random", O_RDONLY);
    if (random_fd == -1) {
        printf("Error opening /dev/random\n");
        return 1;
    }

    Keypair key;
    printf("keypair is %d size\n", sizeof(Keypair));

    ssize_t bytes_read = read(random_fd, &key, sizeof(Keypair));
    if (bytes_read == -1) {
        printf("failed to read from /dev/random\n");
        return 2;
    }

    Keypair *device_key;
    cudaMalloc((void**)&device_key, sizeof(Keypair));
    cudaMemcpy(device_key, &key, sizeof(Keypair), cudaMemcpyHostToDevice);

    search_kernel<<<1, 1>>>(device_key);

    cudaMemcpy(&key, device_key, sizeof(Keypair), cudaMemcpyDeviceToHost);

    printf("public key:\n");
    print_key(key.a, CURVE25519_KEY_SIZE);
    printf("private key:\n");
    print_key(key.b, CURVE25519_KEY_SIZE);
    return 0;
}
