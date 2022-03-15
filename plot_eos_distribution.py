import json
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open("eos_distribution.pkl",'rb') as f:
    length_to_distribution = pickle.load(f)

# for length in range(16, 48):
#     eos_dis_list = [x[0] for x in length_to_distribution[length]]
#     eos_dis_array = np.stack([np.array(x) for x in eos_dis_list], axis = 0)
#     eos_dis_array = np.mean(eos_dis_array, axis = 0)
#     x_list = np.arange(len(eos_dis_array))
#     plt.bar(x = x_list, height = eos_dis_array)
#     plt.savefig("./fig/eos_dis_%d"%(length))
#     plt.close()

for length in range(16, 48):
    # print(length)
    eos_dis_list = [x[1] for x in length_to_distribution[length]]
    # len_list = [len(x[1]) for x in eos_dis_list]
    # print(len_list)
    eos_dis_array = np.stack([np.array(x) for x in eos_dis_list], axis = 0)
    eos_dis_array = np.mean(eos_dis_array, axis = 0)
    x_list = np.arange(len(eos_dis_array))
    plt.bar(x = x_list, height = eos_dis_array)
    plt.savefig("./fig/second_dis_%d"%(length))
    plt.close()
