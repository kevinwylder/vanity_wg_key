#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/types.h>

#include "curve25519.h"
#include "base64.c"

int main() {
    int random_fd = open("/dev/random", O_RDONLY);
    if (random_fd == -1) {
        printf("Error opening /dev/random\n");
        return 1;
    }

    uint8_t pubkey[CURVE25519_KEY_SIZE];
    uint8_t privkey[CURVE25519_KEY_SIZE];

    ssize_t bytes_read = read(random_fd, privkey, CURVE25519_KEY_SIZE);
    if (bytes_read == -1) {
        printf("failed to read from /dev/random\n");
        return 2;
    }

    curve25519_generate_public(pubkey, privkey);
    printf("public key:\n");
    print_key(pubkey, CURVE25519_KEY_SIZE);
    printf("private key:\n");
    print_key(privkey, CURVE25519_KEY_SIZE);
    return 0;
}
