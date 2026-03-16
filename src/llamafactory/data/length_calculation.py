from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_length(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # calculate sample length
    lengths = []
    # Add num_proc=128 here to use 128 processes for processing, effectively improving the processing speed.
    dataset = dataset.map(lambda x: {"length": len(x["input_ids"])}, num_proc=128)  
    lengths = np.array(dataset["length"])

    # for item in tqdm(dataset_module["train_dataset"]):
    #     lengths.append(len(item['input_ids']))
    # lengths = np.array(lengths)
        
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title('Distribution of input_ids lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, 'length_distribution.png'))  # 保存为PNG文件
    plt.close()  # 关闭图形，释放内存
    
    with open(os.path.join(save_dir, 'length_distribution.txt'), 'w', encoding='utf-8') as f:
    # 定义一个函数同时输出到终端和文件
        def print_both(text):
            print(text)  # 输出到终端
            print(text, file=f)  # 输出到文件

        print("*" * 80)
        print_both(f"Sequence length statistics:")
        print_both(f"Shortest length: {min_len}")
        print_both(f"Longest length: {max_len}")
        print_both(f"Average length: {avg_len:.2f}")
        print_both("Length in the range [0, 1k): " + str(sum((lengths >= 0) & (lengths < 1024))))
        print_both("Length in the range [1k, 2k): " + str(sum((lengths >= 1024) & (lengths < 2048))))
        print_both("Length in the range [2k, 3k): " + str(sum((lengths >= 2048) & (lengths < 3072))))
        print_both("Length in the range [3k, 4k): " + str(sum((lengths >= 3072) & (lengths < 4096))))
        print_both("Length in the range [4k, 8k): " + str(sum((lengths >= 4096) & (lengths < 8192))))
        print_both("Length in the range [8k, 12k):   " + str(sum((lengths >= 8192) & (lengths < 12288))))
        print_both("Length in the range [12k, 16k):   " + str(sum((lengths >= 12288) & (lengths < 16384))))
        print_both("Length in the range [16k, ):   " + str(sum(lengths >= 16384)))
        print("*" * 80)