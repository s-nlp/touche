import requests
import sys
import argparse

#numpy
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--i', default='/notebook/Touche22/test/')
    parser.add_argument('--o', default='/notebook/Touche22/test/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    
    output_dir = args.o
    input_dir = args.i
    input_file = args.inp_file
    
    with open('/home/chekalina/Touche22/test/fune_tune_colbert.txt') as file:
        lines = file.readlines()
    
    with open(output_dir + 'run.txt', 'w') as fp:
        for line in lines:
            fp.write(line)
    