#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# String to Binary Stream
def iter2bin(word):
    word_iter = word.encode('ascii')
    return (format(b, '08b') for b in word_iter)

def s2b(word):
    word_bin = list(''.join(s for s in iter2bin(word)))
    word_int = list(map(int, word_bin))
    return word_int

# Binary  stream to string
def binary_stream_to_string(bits):
    chars = []
    for b in range(int(len(bits) / 8)):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

# Creating diagram
def diagram_creator(x, y, title, x_label, y_label):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# Tile reproduction
def tile(value, count):
    return [value for _ in range(int(count))]

# Creating hamming encode array
def ham_code_creator(word_len, step, step2, code, method, *arg):
    code_index = []
    coded_bits = []
    for i in range(0, len(word_len)+1, step):
        code_index.extend([i])
    for z in range(0,len(code_index)-1, step2):
        if z == 22:
            code_method = method(arg, code[code_index[z-1]:code_index[z]])
            coded_bits.extend(code_method)
        else:
            code_method = method(arg, code[code_index[z]:code_index[z+1]])
            coded_bits.extend(code_method)
    return np.hstack(coded_bits)

# Creating hamming decode array
def ham_decode_creator(step, step2, decode):
    decode_index = []
    decoded_bits = []
    for i in range(0, len(decode)+1, step):
        decode_index.extend([i])
    for z in range(0,len(decode_index)-1, step2):
        if z == 22:
            decode_method = hamming_decoder(decode[decode_index[z-1]:decode_index[z]])
            decoded_bits.extend(decode_method)
        else:
            decode_method = hamming_decoder(decode[decode_index[z]:decode_index[z+1]])
            decoded_bits.extend(decode_method)
    return np.hstack(decoded_bits)

# Hamming Encode
def hamming_encoode(g_matrix, array):
    ham_vec = np.dot(g_matrix, array).transpose() % 2
    return ham_vec

#  Hamming Decoder
def hamming_decoder(bits):
    decoded_bits = [None] * 4
    parity_bits = [None] * 3

    parity_bits[0] = (bits[0] + bits[2] + bits[4] + bits[6]) % 2
    parity_bits[1] = (bits[1] + bits[2] + bits[5] + bits[6]) % 2
    parity_bits[2] = (bits[3] + bits[4] + bits[5] + bits[6]) % 2

    n = parity_bits[0] + parity_bits[1] * 2 + parity_bits[2] * 4 - 1
    if n >= 0:
        print('Hamming decoding found error at {} bit'.format(n))
        bits[n] = not bits[n]

    decoded_bits[0] = bits[2]
    decoded_bits[1] = bits[4]
    decoded_bits[2] = bits[5]
    decoded_bits[3] = bits[6]
    return decoded_bits

# Information signal
def information_signal(samples_freq, bits, samples):
    time = list(np.linspace(0, samples_freq * len(bits), samples * len(bits)))

    info_samples = samples * len(bits) * [None]
    for i, bit in enumerate(bits):
        info_samples[i * samples:(i + 1) * samples] = tile(bit, samples)
    return time, info_samples

# CLK
def clock_signal_generator(samples, clk_counts):
    samples_half = int(samples / 2)

    clk_samples = samples * clk_counts * [None]
    for i in range(clk_counts * 2):
        clk_samples[i * samples_half:(i + 1) *
                    samples_half] = tile((i % 2) == 0, samples_half)
    return clk_samples

# Man encode
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

# Man decode
def manchester_decode(clock, manchester, samples):
    clock = tile(1, int(samples / 4)) + clock
    clock = clock[:int(len(clock) - (int(samples / 4)))]

    bits = []
    for i in range(len(clock) - 1):
        if (clock[i] == 1 and clock[i + 1] == 0):
            bits.append(manchester[i])
    return bits

if __name__ == '__main__':
    # 1
    test_word_iter = iter2bin('Ala ma kota')
    test_word_bin = np.array(s2b('Ala ma kota'))

    # 2, 3, 4
    G = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], 
                [ 0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])           
    ham_to_modulation = ham_code_creator(test_word_bin, 4, 1, test_word_bin, hamming_encoode, G)
    samples_frequency = 0.1
    samples = 10000
    clk = clock_signal_generator(samples, len(ham_to_modulation))
    t, ttl = information_signal(samples_frequency, ham_to_modulation, samples)
    encode_manchester = manchester_encode(clk, ttl)
    decode_manchester = np.array(manchester_decode(clk, ttl, samples))
    t_deco, manchester_ttl = information_signal(samples_frequency, decode_manchester, samples)
    ham_decoded_stream = ham_decode_creator(7, 1, decode_manchester)

    # Printing results
    print('String "Ala ma kota" to binary stream: {}.'.format(test_word_bin))
    print('Bit stream after Hamming encoding: {}.'.format(ham_to_modulation))
    diagram_creator(t, encode_manchester, 'Manchester encode', 't[s]', 'Manchester encode')
    diagram_creator(t_deco, manchester_ttl, 'Manchester TTL', 't[s]', 'Manchester TTL')
    print('Stream bits after Manchester decoding: {}.'.format(decode_manchester))
    print('Stream bits after Hamming decoding: {}.'.format(ham_decoded_stream))
    print('Binary stream after Hamming decoding to string: {}.'.format(binary_stream_to_string(ham_decoded_stream)))

#     String "Ala ma kota" to binary stream: [0 1 0 0 0 0 0 1 0 1 1 0 1 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 1 0 1
#     1 0 1 0 1 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 1 0 1 0 1 1 0 1 1 0 1 1 1 1 0 1
#     1 1 0 1 0 0 0 1 1 0 0 0 0 1].
#     Bit stream after Hamming encoding: [1 0 0 1 1 0 0 1 1 0 1 0 0 1 1 1 0 0 1 1 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1 1
#     0 1 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 1 0 1 0 1 0 1 1 1 0 0
#     1 1 0 1 1 0 1 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1
#     1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 1
#     1 0 1 0 0 1].
#     Stream bits after Manchester decoding: [1 0 0 1 1 0 0 1 1 0 1 0 0 1 1 1 0 0 1 1 0 0 1 1 1 1 0 0 1 1 0 0 1 1 0 1 1
#     0 1 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 1 0 1 0 1 0 1 1 1 0 0
#     1 1 0 1 1 0 1 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1
#     1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 1
#     1 0 1 0 0 1].
#     Stream bits after Hamming decoding: [0 1 0 0 0 0 0 1 0 1 1 0 1 1 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 1 0 1
#     1 0 1 0 1 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 1 0 1 0 1 1 0 1 1 0 1 1 1 1 0 1
#     Binary stream after Hamming decoding to string: Ala ma kota.