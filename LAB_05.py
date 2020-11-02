#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# FEDCBA
# 026411

# Creating diagram
def diagram_creator(x, y, title, x_label, y_label):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# subplot creator
def subplot_creator(t1, sub1, title1, x1, y1, t2, sub2, title2, x2, y2, t3, sub3, title3, x3, y3,):
    plt.figure(figsize=(16, 9))
    sub = plt.subplot(3, 1, 1)
    sub.plot(t1, sub1)
    plt.title(title1)
    plt.xlabel(x1)
    plt.ylabel(y1)
    sub = plt.subplot(3, 1, 2)
    sub.plot(t2, sub2)
    plt.title(title2)
    plt.xlabel(x2)
    plt.ylabel(y2)
    sub = plt.subplot(3, 1, 3)
    sub.plot(t3, sub3)
    plt.title(title3)
    plt.xlabel(x3)
    plt.ylabel(y3)
    plt.show()

# String to Binary Stream
def string_to_binary_stream(slowo):
    result = []
    for c in slowo:
        bits = bin(ord(c))[2:]
        bits = '0000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

# ASK
def amplitude_shift_keying(info_signal, t, a1, a2):
    ask = []
    for i, j in zip(info_signal, t):
        if i == 1:
            ask.append(a1 * np.sin(2 * np.pi * j * f + fi))
        if i == 0:
            ask.append(a2 * np.sin(2 * np.pi * j * f + fi))
    return ask

# FSK
def frequency_shift_keying(info_signal, t, a1, a2, f0, f1):
    fsk = []
    for i, j in zip(info_signal, t):
        if i == 1:
            fsk.append(a1 * np.sin(2 * np.pi * j * f1 + fi))
        if i == 0:
            fsk.append(a1 * np.sin(2 * np.pi * j * f0 + fi))
    return fsk

# PSK
def phase_shift_keying(info_signal, t, a1, a2):
    psk = []
    for i, j in zip(info_signal, t):
        if i == 1:
            psk.append(a1 * np.sin(2 * np.pi * j * 1 + fi))
        if i == 0:
            psk.append(a1 * np.sin(2 * np.pi * j * 1 + 0))
    return psk

# Discrete Fourier Transform
def dft(x):
    N = len(x)
    Xk = []
    wN = np.exp((1j * 2 * np.pi) / N)
    for k in range(N):
        sum_dft = 0
        for n in range(N):
            sum_dft = sum_dft + x[n] * wN ** (-k * n)
        Xk.append(sum_dft)
    return Xk

def bandwidth(z):
    f_min = np.min(z)
    f_max = np.max(z)
    w = f_max - f_min
    print(w)

if __name__ == '__main__':
    # Task 1
    test_word = string_to_binary_stream('test')
    print(test_word)
    # 01110100011001010111001101110100

    # Task2
    tb = 1
    end = len(test_word)
    sample_number = 50
    samples = sample_number * (end / tb)
    t = np.linspace(0, end, int(samples))
    x = np.linspace(0, end, end)
    interpolation = interp1d(x, test_word, kind='previous')
    info_signal = interpolation(t)
    diagram_creator(t, info_signal, 'Information signal for test word', 'Information signal', 't[s]')

    amplitude1 = 0.1
    amplitude2 = 1
    n = 1/tb
    f = n * (tb ** -1)
    fi = np.pi
    f0 = (n + 1)/tb
    f1 = (n + 2)/tb
    ask = amplitude_shift_keying(info_signal, t, amplitude1, amplitude2)
    fsk = frequency_shift_keying(info_signal, t, amplitude1, amplitude2, f0, f1)
    psk = phase_shift_keying(info_signal, t, amplitude1, amplitude2)
    subplot_creator(t, ask, 'Amplitude shift keying', 'Time [s]', 'Amplitude', 
                    t, fsk, 'Frequency shift keying', 'Time [s]', 'Amplitude', 
                    t, psk, 'Phase shift keying', 'Time [s]', 'Amplitude')

    #Task3
    n = 2
    t2 = np.linspace(0, 10, int(samples))
    x2 = np.linspace(0, 10, end)
    interpolation = interp1d(x2, test_word, kind='previous')
    info_signal2 = interpolation(t2)
    ask2 = amplitude_shift_keying(info_signal2, t2, amplitude1, amplitude2)
    fsk2 = frequency_shift_keying(info_signal2, t2, amplitude1, amplitude2, f0, f1)
    psk2 = phase_shift_keying(info_signal2, t2, amplitude1, amplitude2)
    subplot_creator(t2, ask2, 'Amplitude shift keying, N=2, 10 bits length', 'Time [s]', 'Amplitude', 
                    t2, fsk2, 'Frequency shift keying, N=2, 10 bits length','Time [s]', 'Amplitude', 
                    t2, psk2, 'Phase shift keying, N=2, 10 bits length', 'Time [s]', 'Amplitude')
    
    #Task4
    ask_dft = dft(ask)
    fsk_dft = dft(fsk)
    psk_dft = dft(psk)
    ask_spectrum = np.abs(ask_dft)
    fsk_spectrum = np.abs(fsk_dft)
    psk_spectrum = np.abs(psk_dft)
    subplot_creator(t, ask_spectrum, 'Amplitude spectrum - ASK','Frequency','Decibels',
                    t, fsk_spectrum, 'Amplitude spectrum - FSK','Frequency','Decibels',
                    t, psk_spectrum, 'Amplitude spectrum - PSK','Frequency','Decibels',)
    
    #Task5
    bandwidth(ask) #1.9999785685132996
    bandwidth(fsk) #0.19999987393216945
    bandwidth(psk) #0.19999987393216942