from itertools import count
import matplotlib.pyplot as plt
import pickle
import numpy as np

def visualize_lengbias():
    with open("./saved_results/distance_transformer.pkl", 'rb') as f:
        l1_distance_dict, l2_distance_dict = pickle.load(f)

    keys = list(l1_distance_dict.keys())
    keys.sort()
    keys = np.array(keys)

    l1_mean_bias = np.array([np.mean(l1_distance_dict[x]) for x in keys])
    l2_mean_bias = np.array([np.mean(l2_distance_dict[x]) for x in keys])

    plt.bar(x = keys, height = l2_mean_bias)
    plt.savefig("l2_bias.png")
    plt.close('all')

    plt.bar(x = keys, height = l1_mean_bias)
    plt.savefig("l1_bias.png")


def visualize_recall(beam_size = 5): 
    with open("./saved_results/recall_beam%d_26-35"%(beam_size), 'rb') as f:
        dict1 = pickle.load(f)
    with open("./saved_results/recall_beam%d_36-48"%(beam_size), 'rb') as f:
        dict2 = pickle.load(f)
    if beam_size == 5:
        dict3 = {
            16: [15413, 17013],
            17: [11283, 16873],
            18: [8288, 16362],
            19: [6625, 16659],
            20: [5661, 16536],
        }
    else:
        with open("./saved_results/recall_beam%d_16-20"%(beam_size), 'rb') as f:
            dict3 = pickle.load(f)
    with open("./saved_results/recall_beam%d_21-25"%(beam_size), 'rb') as f:
        dict4 = pickle.load(f)



    recall_list = np.zeros(50)
    for k,v in dict1.items():
        recall_list[k] = dict1[k][0] / dict1[k][1]
    for k,v in dict2.items():
        recall_list[k] = dict2[k][0] / dict2[k][1]
    for k,v in dict3.items():
        recall_list[k] = dict3[k][0] / dict3[k][1]
    for k,v in dict4.items():
        recall_list[k] = dict4[k][0] / dict4[k][1]

    keys = np.array([i for i in range(16,48)])
    values = recall_list[16:48]
    plt.bar(x = keys, height = values)
    # plt.plot(keys, values)
    plt.savefig("saved_results/recall_beam%d.png"%(beam_size))


def visualize_probability(start = 16, end = 20):
    with open("./saved_results/probability_dist_%d-%d.pkl"%(start, end), 'rb') as f:
        before_reweight, after_reweight = pickle.load(f)
    keys = np.array([i+1 for i in range(end)])
    print(keys)
    print(len(before_reweight))
    plt.bar(x = keys, height = before_reweight, alpha=0.5)
    plt.savefig('saved_results/before_probability_dist_%d-%d'%(start, end))
    plt.close('all')

    plt.bar(x = keys, height = after_reweight, alpha=0.5)
    plt.show()
    plt.savefig('saved_results/after_probability_dist_%d-%d'%(start, end))

def extract_log(file_addr):
    with open(file_addr, 'r', encoding = 'utf-8') as f:
        train_iter_idx_list = []
        valid_iter_idx_list = []
        train_mc_acc_list = []
        train_adv_acc_list = []
        valid_mc_acc_list = []
        valid_adv_acc_list = []
        for line in f:
            content = line.strip()
            if "Iteration" in content:
                iter_idx = int(content[27:35])
                mc_acc = float(content[61:66])
                adv_acc = float(content[79:84])
                train_iter_idx_list.append(iter_idx)
                train_mc_acc_list.append(mc_acc)
                train_adv_acc_list.append(adv_acc)

            elif "Evaluation Accuracy" in content:
                mc_acc = float(content[40:45])
                adv_acc = float(content[103:108])
                valid_iter_idx_list.append(iter_idx)
                valid_mc_acc_list.append(mc_acc)
                valid_adv_acc_list.append(adv_acc)

    return train_iter_idx_list, valid_iter_idx_list, train_mc_acc_list, train_adv_acc_list, valid_mc_acc_list, valid_adv_acc_list

def visualize_training_curve(file_addr):
    train_iter_idx_list, valid_iter_idx_list, train_mc_acc_list, train_adv_acc_list, valid_mc_acc_list, valid_adv_acc_list = extract_log(file_addr)

    # plt.plot(train_iter_idx_list, train_mc_acc_list, label = 'train_mc')

    train_iter_idx_list, train_adv_acc_list = transform_train_data(train_iter_idx_list, train_adv_acc_list, valid_iter_idx_list)
    plt.plot(train_iter_idx_list, train_adv_acc_list, label = 'train_adv')

    # plt.plot(valid_iter_idx_list, valid_mc_acc_list, label = 'valid_mc')
    plt.plot(valid_iter_idx_list, valid_adv_acc_list, label = 'valid_adv')

    plt.xlabel("iteration")
    plt.ylabel("accuracy")

    plt.legend()
    plt.savefig("./saved_results/fig.png")


def transform_train_data(train_iter_idx_list, train_adv_acc_list, valid_iter_idx_list):
    transformed = []
    count = 0
    prev_list = []
    for idx in range(len(train_iter_idx_list)):
        if count >= len(valid_iter_idx_list):
            break
        # print(idx, count, len(valid_iter_idx_list))
        # print(train_iter_idx_list[idx])
        # print(valid_iter_idx_list[count])
        if train_iter_idx_list[idx] != valid_iter_idx_list[count]:
            prev_list.append(train_adv_acc_list[idx])
        else:
            count += 1
            transformed.append(np.mean(prev_list))
            prev_list = []
    return valid_iter_idx_list, transformed

def plot_sanity_check():
    # target_logits = [0.0232, -0.3756, 0.5142, 0.8885, 0.9302, 1.5243, -0.5402, 1.7811]
    # other_logits = [0.097, 0.1093, 0.4887, 0.8876, 1.4598, 1.6624, 0.5114, 0.9610]

    target_logits = [-0.1463, 1.8,  3.1005, 4.0222]
    other_logits = [0.1223, -0.16,  -0.3551, -2.2126]


    x_value = [i for i in range(len(target_logits))]
    plt.plot(x_value, target_logits, label = 'logits_target')
    plt.plot(x_value, other_logits, label = 'logits_other')

    plt.xlabel("iteration")
    plt.ylabel("logits")

    plt.legend()
    plt.savefig("./saved_results/fig.png")




if __name__ == '__main__':
    # visualize_probability(16,20)
    # visualize_probability(21,25)
    # visualize_probability(26,35)
    # visualize_probability(36,48)

    # visualize_recall(5)

    # extract_log(file_addr = './checkpoints/mc_easycls_adv_only/train_log.txt')
    # visualize_training_curve('./checkpoints/mc_adv_only/train_log.txt')

    # visualize_training_curve('./checkpoints/mc_easycls_adv_only/train_log.txt')
    # visualize_training_curve('./checkpoints/mc_easycls_gan/train_log.txt')


    # visualize_training_curve('./checkpoints/gan_cls_1.0/train_log.txt')

    plot_sanity_check()
