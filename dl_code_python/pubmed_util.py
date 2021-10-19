import torch
import os
import numpy as np
import torch
import scipy.sparse as sp


def read_index_info(file_path):
    crs = open(file_path, "r")
    #print (crs.read())
    a = [line.split() for line in crs]
    result = []
    for each in a:
#         print(each)
        result.append(int(each[0]))
    
#     # convert the edge to src and dst
#     src = []
#     dst = []
#     for each_edge in array:
#         src.append(int(each_edge[0]))
#         dst.append(int(each_edge[1]))
    return result



def read_label_info(file_path):
    crs = open(file_path, "r")
#     print ("Output of Read function is ")
#     #print (crs.read())
    a = [line.split() for line in crs]
    result = []
    for each in a:
        result.append(int(each[0]))
    return result




def read_feature_info(file_path):
    crs = open(file_path, "r")
#     print ("Output of Read function is ")
#     #print (crs.read())
    a = [line.split() for line in crs]
#     print(a[0])
    result = []
    for each in a:
        temp = []
#         print(each)
        for each_ele in each:
#             print(each_ele)
            temp.append(float(each_ele))
        result.append(temp)
    return result


def accuracy(output, labels):
    # preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    # correct = correct.sum()
    correct = 0
    predict = torch.max(output, 1).indices
    # print("emmmm?")
    # print(predict)
    # print(labels)
    for i in range(len(labels)):
        if (predict[i] == labels[i]):
            correct = correct + 1
    return correct / len(labels)

