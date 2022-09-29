#!/usr/bin/python3
#
# @file converter.py
# @author Nikolaus Binder, NVIDIA
# @description Converts a matlab data file to a binary file that can be used
#              with the loader in binary_file_reader.h
#              Requires sio, numpy, and scipy

from array import array
import scipy.io as sio
import sys
import numpy

if len(sys.argv) < 2:
    print("Usage: converter.py [file1.mat] [file2.mat] ...")
    exit(1)
else:
    input_files = sys.argv
    input_files.pop(0)

def get_sizes(data):
    lengths = []
    lengths.append(len(data))
    if hasattr(data[0], "__len__"):
        lengths += get_sizes(data[0])
    elif isinstance(data[0], complex) or isinstance(data[0], numpy.complex64):
        lengths += [2]
    return lengths

def print_data(m, output_file):
    if hasattr(m, "__len__"):
        for i in m:
            print_data(i, output_file)
    elif isinstance(m, complex) or isinstance(m, numpy.complex64):
        array('f', [float(m.real), float(m.imag)]).tofile(output_file)
    else:
        array('f', [m]).tofile(output_file)

for input_filename in input_files:
    output_filename = input_filename.replace('.mat', '.bin')
    input_data = sio.loadmat(input_filename)
    items = []
    for key, item in input_data.items():
        if not key.startswith("__"):
            sizes = get_sizes(input_data[key])
            items.append((key, sizes))
    print(items)
    with open(output_filename, 'wb') as output_file:
        array('i', [len(items)]).tofile(output_file) # number of keys
        for item in items:
            key_bytes = array('b')
            key_bytes.frombytes(item[0].encode())
            array('i', [len(key_bytes)]).tofile(output_file)
            key_bytes.tofile(output_file)
        for item in items:
            array('i', [len(item[1])]).tofile(output_file)
            array('i', item[1]).tofile(output_file)
            print_data(input_data[item[0]], output_file)
