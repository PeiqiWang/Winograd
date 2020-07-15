from winograd import cookToomFilter, winograd
import argparse
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--r', type=int, default=3, help='the width of filter')
parser.add_argument('--n', type=int, default=2, help='the width of input tile')
parser.add_argument('--s', type=int, default=1, help='stride')

# no padding
args = parser.parse_args()
input_size = args.n
kernel_size = args.r
stride = args.s
output_size = input_size - kernel_size + 1
input_channel = 1
output_channel = 1

filters = np.random.normal(0.0,0.1, (input_channel,output_channel,kernel_size,kernel_size))
input_data = np.random.uniform(-1.0,1.0, (input_channel,output_channel,input_size,input_size))

# direct conv
output_d = F.conv2d(torch.tensor(input_data), torch.tensor(filters), stride=stride)

#winograd
output_w = winograd(input_data, filters, output_size, kernel_size, stride)

output_d = output_d.numpy().flatten()
output_w = output_w.flatten()

if len(output_d) != len(output_w):
	print("Error! The lengths of results with two methods are different.")

correct = True
for i in range(len(output_d)):
	if (abs(output_w[i]-output_d[i])/output_d[i] > 1e-4):
		correct = False
		print("Error! winograd[", i,"]=",output_w[i],"||directConv[",i,"]=",output_d[i])
		break
if correct:
	print("Output correct.")
