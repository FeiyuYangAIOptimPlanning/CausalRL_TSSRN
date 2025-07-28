# MIT License

# Copyright (c) 2024 Feiyu Yang 杨飞宇

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_vectors(tensor1, tensor2, file_path):
    """

    """
    num_vectors=16
    pred = tensor1.clone().detach().cpu().numpy()
    true = tensor2.clone().detach().cpu().numpy()

    y_min = min(np.min(pred[:num_vectors]), np.min(true[:num_vectors]))
    y_max = max(np.max(pred[:num_vectors]), np.max(true[:num_vectors]))

    rows = int(np.sqrt(num_vectors))
    cols = (num_vectors + rows - 1) // rows

    fig, axs = plt.subplots(rows, cols, figsize=(15, 10), sharey=True) 
    axs = axs.flatten() 
    for i in range(num_vectors):
        ax = axs[i]
        ax.plot(pred[i], label='Predicted', marker='o')
        ax.plot(true[i], label='True', marker='x')
        ax.set_xlabel('Reward dimension')
        ax.set_ylabel('Value')
        ax.set_ylim(y_min, y_max)
        ax.legend()

    plt.tight_layout()

    plt.savefig(file_path)
    plt.show()


def plot_combined_error_boxplots(pred_tensors, true_tensors, file_path):
    """
    
    """
    assert len(pred_tensors) == len(true_tensors) == 18, "The length of the tensors list must be 18."

   
    error_data = []
    for i in range(18):
      
        error = (torch.mean(pred_tensors[i], dim=0) - torch.mean(true_tensors[i], dim=0)).cpu().numpy().flatten()
        error_data.append(error)

    
    plt.figure(figsize=(15, 10))

   
    plt.boxplot(error_data, showfliers=False)
    plt.xlabel('Agent index')
    plt.ylabel('Error')
    plt.xticks(range(1, 19), range(1, 19))  


    plt.savefig(file_path)

    plt.show()

def plot_combined_error_violinplots(pred_tensors, true_tensors, file_path):
    """
 
    """
    assert len(pred_tensors) == len(true_tensors) == 18, "The length of the tensors list must be 18."


    error_data = []
    for i in range(18):
        
        error = (torch.mean(pred_tensors[i], dim=0) - torch.mean(true_tensors[i], dim=0)).cpu().numpy().flatten()
        error_data.append(error)


    plt.figure(figsize=(15, 10))


    plt.violinplot(error_data)
    plt.xlabel('Agent index')
    plt.ylabel('Error')
    plt.xticks(range(1, 19), range(1, 19)) 


    plt.savefig(file_path)
 
    plt.show()

