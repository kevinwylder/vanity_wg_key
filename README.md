# Vanity Wireguard Key Generator

Find interesting wireguard keys

If you have multiple devices using wireguard, it can be nice to generate keys with identifyable
prefixes to differentiate clients, as wireguard does not have an existing way to name clients.

This program works with brute force, so is optimized as much as possible to reduce the energy
impact of searching for a vanity key.

## Requirements

* nvidia GPU
* `nvcc` from the nvidia toolchain

## Recommendations

Underclocking the GPU can have a significant improvement in hashes per watt. I found this was
optimal on my 3080

```
nvidia-smi -pl 150
```

It is important to get a wholistic view of the power consumption for getting the optimal hashes
per watt, a home outlet power meter can compute the entire rig energy.

## How to set a key prefix

The program calls `match` on every key, and terminates once a key is found that returns `true`

By default, the program creates a key that starts with `kwylder` (me!). To choose a different
prefix, you'll need to pick bytes that base64 encode to your target sequence.

Note that base64 characters only encode 6 bits, so you may need to pad your search term to find
the range of values that are accepted. Pad with `A` for the lower bound, and `/` for the upper

```
[kwylder@computer:~/vanitykey]$ echo 'kwylderA' | base64 -d | hexdump -C
00000000  93 0c a5 75 ea c0                                 |...u..|
00000006
[kwylder@computer:~/vanitykey]$ echo 'kwylder/' | base64 -d | hexdump -C
00000000  93 0c a5 75 ea ff                                 |...u..|
00000006
```

With these byte sequences, modify the `match` function with your prefix

