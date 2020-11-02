#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Creating diagram
def diagram_creator(x, y, title, x_label):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.show()

#tile reproduction
def tile(value, count):
    return [value for _ in range(int(count))]

#CLK
def clock_signal_generator(samples, clk_counts):
    samples_half = int(samples / 2)

    clk_samples = samples * clk_counts * [None]
    for i in range(clk_counts * 2):
        clk_samples[i * samples_half:(i + 1) * samples_half] = tile((i % 2) == 0, samples_half)
    return clk_samples

#TTL
def information_signal(samples_freq, bits, samples):
    time = list(np.linspace(0, samples_freq * len(bits), samples * len(bits)))

    info_samples = samples * len(bits) * [None]
    for i, bit in enumerate(bits):
        info_samples[i * samples:(i + 1) * samples] = tile(bit, samples)
    return time, info_samples

#Man encode
def manchester_encode(clk, ttl):
    encode = [0]
    val = 0

    for i in range(len(clk) - 1):
        if (clk[i] == 1 and clk[i + 1] == 0):
            if (ttl[i] == 0):
                val = 1
            else:
                val = -1
        elif (clk[i] == 0 and clk[i + 1] == 1):
            if (ttl[i] == ttl[i + 1]):
                val *= -1
        encode.append(val)
    return encode

#Man decode
def manchester_decode(clock, manchester, samples):
    clock = tile(1, int(samples / 4)) + clock
    clock = clock[:int(len(clock) - (int(samples / 4)))]

    bits = []
    for i in range(len(clock) - 1):
        if (clock[i] == 1 and clock[i + 1] == 0): 
            bits.append(manchester[i])
    return bits

if __name__ == '__main__':
    #input data
    samples_frequency = 0.1
    samples = 50
    bits = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]

    clk = clock_signal_generator(samples, len(bits))
    t, ttl = information_signal(samples_frequency, bits, samples)
    encode_manchester = manchester_encode(clk, ttl)
    decode_manchester = manchester_decode(clk, ttl, samples)
    t_deco, manchester_ttl = information_signal(samples_frequency, decode_manchester, samples)

    diagram_creator(t, encode_manchester, 'Manchester encode', 't[s]')
    diagram_creator(t_deco, manchester_ttl, 'Manchester TTL', 't[s]')
    diagram_creator(t, clk, 'CLK', 't[s]')
