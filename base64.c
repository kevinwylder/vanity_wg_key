#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

static const unsigned char base64_table[65] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

void print_key(const uint8_t *buf, size_t size) {
    const uint8_t *end = buf + size;
    const uint8_t *in = buf;
    while (end - in >= 3) {
        printf(
            "%c%c%c%c",
            base64_table[in[0] >> 2],
            base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)],
            base64_table[((in[1] & 0x0f) << 2) | (in[2] >> 6)],
            base64_table[in[2] & 0x3f]
        );
        in += 3;
    }
    if (end - in) {
        printf("%c", base64_table[in[0] >> 2]);
        if (end - in == 1) {
            printf("%c=", base64_table[(in[0] & 0x03) << 4]);
        }
        else {
            printf(
                "%c%c", 
                base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)], 
                base64_table[(in[1] & 0x0f) << 2]
            );
        }
        printf("=\n");
    } else {
        printf("\n");
    }
}

