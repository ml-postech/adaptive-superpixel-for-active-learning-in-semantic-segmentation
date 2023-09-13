import os
import copy
import math
import pickle
import random
import argparse
import cv2 as cv
import numpy as np
import scipy.special
import scipy.ndimage
import networkx as nx
import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt


from skimage import exposure
from PIL import Image, ImageDraw
from matplotlib.colors import ListedColormap
from skimage.segmentation import slic, mark_boundaries
from multiprocessing import Semaphore
from scipy.spatial import distance


_IGNORE_LABEL = 255
INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'ResizeBilinear_2:0'
os.environ["CUDA_VISIBLE_DEVICES"]="2"


eps = 0.1
save_job_name = 'adaptive_spx_r1'
model_job_name = '8192_50k_v0_1'
batch_id = 0
superpixel_label_dir = './superpixels/cityscapes/seeds_8192/train/label'
superpixel_result_dir = './superpixels/cityscapes/' + save_job_name + '_' + str(eps) + '/batch_' + str(batch_id)
if not os.path.exists(superpixel_result_dir + '/result'):
    os.makedirs(superpixel_result_dir + '/label')
    os.makedirs(superpixel_result_dir + '/result')


def build_region_graph(feature, R):
  feature_sorted = np.sort(feature, axis=2)
  uncertainty = feature_sorted[:,:,-2] / feature_sorted[:,:,-1]

  # Region Color, Region Feature
  region_indexs = np.unique(R)
  region_feature = {}
  for index in region_indexs:
    y, x = np.where(R == index)
    region_feature[index] = (np.mean(feature[y, x], axis=0), len(y), np.mean(uncertainty[y, x]))

  # Generate Graph
  G = nx.Graph()
  def center(index):
    y, x = np.where(R == index)
    return y[0], x[0] # check !!
  G.add_nodes_from([(index, {"center": center(index)}) for index in region_indexs])

  coords = np.where(R[:, :-1] != R[:, 1:])
  for y, x in zip(*coords):
    u, v = R[y, x], R[y, x + 1]
    if (u, v) not in G.edges(): 
        G.add_edge(u, v)

  coords = np.where(R[:-1, :] != R[1:, :])
  for y, x in zip(*coords):
    u, v = R[y, x], R[y + 1, x]
    if (u, v) not in G.edges():
        G.add_edge(u, v)
  
  return G, region_feature


def merge_graph_from_uncert(G, R, feature):
    unc = {}
    for u in G.nodes():
        unc[u] = feature[u][2]
    sorted_unc = {k : v for k, v in sorted(unc.items(), key=lambda item : -item[1])} # check !!

    merged_arr = []
    for u in sorted_unc.keys():
        if u not in merged_arr:
            merged_arr.append(u)
            
            adj_arr = []
            for v in G.neighbors(u):
                if v not in merged_arr:
                    org_idx = R[G.nodes[u]['center']]
                    merged_idx = R[G.nodes[v]['center']]
                    if org_idx != merged_idx:
                        if distance.jensenshannon(feature[u][0], feature[v][0]) < eps:
                            R = np.where(R == merged_idx, org_idx, R)
                            feature[v] = feature[u]
                            merged_arr.append(v)
                            adj_arr.append(v)

            while adj_arr != []:
                adj = adj_arr.pop()
                for v in G.neighbors(adj):
                    if v not in merged_arr:
                        org_idx = R[G.nodes[adj]['center']]
                        merged_idx = R[G.nodes[v]['center']]
                        if org_idx != merged_idx:
                            if distance.jensenshannon(feature[adj][0], feature[v][0]) < eps:
                                R = np.where(R == merged_idx, org_idx, R)
                                feature[v] = feature[adj]
                                merged_arr.append(v)
                                adj_arr.append(v)
    return R


def merge_graph_from_uncert_complexity(G, R, feature):
    unc = {}
    for u in G.nodes():
        unc[u] = feature[u][2]
    sorted_unc = {k : v for k, v in sorted(unc.items(), key=lambda item : -item[1])} # check !!

    merged_arr = []
    for u in sorted_unc.keys():
        if u not in merged_arr:
            merged_arr.append(u)
            if len(merged_arr) >= 8192 * 0.1:
                break
            
            adj_arr = []
            for v in G.neighbors(u):
                if v not in merged_arr:
                    org_idx = R[G.nodes[u]['center']]
                    merged_idx = R[G.nodes[v]['center']]
                    if org_idx != merged_idx:
                        if distance.jensenshannon(feature[u][0], feature[v][0]) < eps:
                            R = np.where(R == merged_idx, org_idx, R)
                            feature[v] = feature[u]
                            merged_arr.append(v)
                            if len(merged_arr) >= 8192 * 0.1:
                                break
                            adj_arr.append(v)

            while adj_arr != []:
                adj = adj_arr.pop()
                for v in G.neighbors(adj):
                    if v not in merged_arr:
                        org_idx = R[G.nodes[adj]['center']]
                        merged_idx = R[G.nodes[v]['center']]
                        if org_idx != merged_idx:
                            if distance.jensenshannon(feature[adj][0], feature[v][0]) < eps:
                                R = np.where(R == merged_idx, org_idx, R)
                                feature[v] = feature[adj]
                                merged_arr.append(v)
                                if len(merged_arr) >= 8192 * 0.1:
                                    break
                                adj_arr.append(v)
    return R


def region_adj(image_name_path, sess):
    image_name, image_path = image_name_path

    # if image_name not in image_name_list:
    #     return

    parts = image_name.split("_")
    image = Image.open(image_path)
    height, width = image.size
    img = image.convert('RGB').resize((height // 2, width // 2))
    img_org = image.convert('RGB').resize((height, width))
    img_eq = exposure.equalize_hist(np.asarray(img_org))

    logits = sess.run( OUTPUT_TENSOR_NAME, feed_dict={INPUT_TENSOR_NAME : [np.asarray(img)]})
    logits = np.squeeze(logits)
    softmax_output = scipy.special.softmax(logits, axis=2)
    feature = scipy.ndimage.zoom(softmax_output, (2, 2, 1), order=0)
    logits = scipy.ndimage.zoom(logits, (2, 2, 1), order=0)

    sp_dic = pickle.load(open(os.path.join(superpixel_label_dir, image_name + '.pkl'), 'rb'))
    sp_labels = sp_dic['labels'] # (1024, 2048)

    G, region_feature = build_region_graph(feature, sp_labels)
    merged_labels_ft = merge_graph_from_uncert(G, sp_labels, region_feature)
    # merged_labels_ft = merge_graph_from_uncert_complexity(G, sp_labels, region_feature)
    print("Merged : 8192 ->", np.unique(merged_labels_ft).shape[0])

    output_dic = {}
    merged_labels_ft = merged_labels_ft.astype(np.int16)
    result = mark_boundaries(img_eq, merged_labels_ft, color=(1,0,0))
    output_dic['labels'] = merged_labels_ft
    output_dic['valid_idxes'] = np.unique(merged_labels_ft)
    pickle.dump(output_dic, open(os.path.join(superpixel_result_dir, 'label/' + image_name + '.pkl'), 'wb'))
    plt.imsave(os.path.join(superpixel_result_dir, 'result/' + image_name + '.jpg'), result)

    return

if __name__ == '__main__':
    devkit_path = './deeplab/datasets/cityscapes/'
    image_dir = devkit_path + 'leftImg8bit/'
    imageset_path = devkit_path + 'image_list/train.txt'

    with open(imageset_path, 'r') as f:
         lines = f.readlines()
    image_list = [x.strip() for x in lines]

    image_name_path = {}
    for image_name in image_list:
      parts = image_name.split("_")
      image_path = os.path.join(image_dir, 'train', parts[0], image_name + '_leftImg8bit.png')
      image_name_path[image_name] = image_path   
    image_name_path = list(image_name_path.items())

    PATH_TO_FROZEN_GRAPH = os.path.join('outputs', model_job_name, 'batch_{}'.format(batch_id), 'frozen_inference_graph.pb')
    seg_graph = tf.Graph()
    with seg_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with seg_graph.as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            for each in image_name_path:
                region_adj(each, sess)
