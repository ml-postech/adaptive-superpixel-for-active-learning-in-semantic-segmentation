import os
import pickle
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import exposure


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='cityscapes')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--sp_method', type=str, default='seeds')
parser.add_argument('--num_superpixels', type=int, default=8192)
parser.add_argument('--resize_factor', type=int, default=1)
FLAGS = parser.parse_args()


def extract_superpixel_slic(image_name_path): 
    method = 'slic'
    
    num_superpixels = FLAGS.num_superpixels
    sigma = 0
    superpixel_label_dir = './superpixels/{}/{}_{}/{}/label'.format(FLAGS.dataset_name, method, num_superpixels, FLAGS.split)
    if not os.path.exists(superpixel_label_dir):
       os.makedirs(superpixel_label_dir)
    superpixel_result_dir = './superpixels/{}/{}_{}/{}/result'.format(FLAGS.dataset_name, method, num_superpixels, FLAGS.split)
    if not os.path.exists(superpixel_result_dir):
       os.makedirs(superpixel_result_dir)
    
    max_n = 0
    nr_sample = 0
    for image_name, image_path in image_name_path.items():
        
        print(image_name) 
        #if os.path.exists(os.path.join(superpixel_label_dir, image_name + '.pkl')):
        #    continue
        img = plt.imread(image_path)
        img_eq = exposure.equalize_hist(img)
        img_eq = img
        
        labels = slic(img_eq, n_segments = num_superpixels, sigma=sigma)
        result = mark_boundaries(img, labels)
        
        output_dic = {}
        output_dic['labels'] = labels.astype(np.int16)
        num_sp = labels.max() + 1

        if num_sp > max_n:
            max_n = num_sp

        output_dic['valid_idxes'] = np.unique(labels)

        pickle.dump(output_dic, open(os.path.join(superpixel_label_dir, image_name + '.pkl'), 'wb'))
        plt.imsave(os.path.join(superpixel_result_dir, image_name + '.jpg'), result)
        nr_sample += 1
    print(max_n)
    
    
def extract_superpixel_seeds(image_name_path): 
    method = 'seeds'

    prior = 3
    num_levels = 5
    num_histogram_bins = 10
    num_superpixels = FLAGS.num_superpixels
       
    superpixel_label_dir = './superpixels/{}/{}_{}/{}/label'.format(FLAGS.dataset_name, method, num_superpixels, FLAGS.split)
    if not os.path.exists(superpixel_label_dir):
       os.makedirs(superpixel_label_dir)
    superpixel_result_dir = './superpixels/{}/{}_{}/{}/result'.format(FLAGS.dataset_name, method, num_superpixels, FLAGS.split)
    if not os.path.exists(superpixel_result_dir):
       os.makedirs(superpixel_result_dir)

    max_n = 0
    nr_sample = 0
    for image_name, image_path in image_name_path.items():
        #if nr_sample >= 100: break
        print(image_name)          
            
        img = Image.open(image_path)
        width, height = img.size
        resize_factor = FLAGS.resize_factor
        img = img.convert('RGB').resize((width//resize_factor, height//resize_factor))
           
        img_eq = exposure.equalize_hist(np.asarray(img))
        #img_eq = img
        
        converted_img = cv.cvtColor(img_eq.astype(np.float32), cv.COLOR_RGB2HSV)
        height,width,channels = converted_img.shape
        seeds = cv.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins, True)
        seeds.iterate(converted_img, 10)

        labels = seeds.getLabels()
        result = mark_boundaries(img_eq, labels)
        
        output_dic = {}
        output_dic['labels'] = labels.astype(np.int16)
        num_sp = labels.max() + 1

        if num_sp > max_n:
            max_n = num_sp

        output_dic['valid_idxes'] = np.unique(labels)

        pickle.dump(output_dic, open(os.path.join(superpixel_label_dir, image_name + '.pkl'), 'wb'))
        plt.imsave(os.path.join(superpixel_result_dir, image_name + '.jpg'), result)
        nr_sample += 1
    print(max_n)
    
if __name__ == '__main__':
    
    if FLAGS.dataset_name == 'pascal_voc_seg':
      devkit_path = './deeplab/datasets/pascal_voc_seg/VOCdevkit/'
      image_dir = devkit_path + 'VOC2012/JPEGImages'
      imageset_path = devkit_path + 'VOC2012/ImageSets/Segmentation/%s.txt'%FLAGS.split
    elif FLAGS.dataset_name == 'cityscapes':
      devkit_path = './deeplab/datasets/cityscapes/'
      image_dir = devkit_path + 'leftImg8bit/'
      imageset_path = devkit_path + 'image_list/%s.txt'%FLAGS.split

    with open(imageset_path, 'r') as f:
         lines = f.readlines()
    image_list = [x.strip() for x in lines]

    image_name_path = {}
 
    if FLAGS.dataset_name == 'pascal_voc_seg':
       for image_name in image_list: 
           image_path = os.path.join(image_dir, image_name + '.jpg')
           image_name_path[image_name] = image_path  
    elif FLAGS.dataset_name == 'cityscapes':
       for image_name in image_list:
          parts = image_name.split("_")
          image_path = os.path.join(image_dir, FLAGS.split, parts[0], image_name + '_leftImg8bit.png')
          image_name_path[image_name] = image_path   
    
    if FLAGS.sp_method == 'seeds':
       extract_superpixel_seeds(image_name_path)
    elif FLAGS.sp_method == 'slic':
       extract_superpixel_slic(image_name_path)
    else:
        print('%s not implemented'%FLAGS.sp_method)
        raise RuntimeError
    
    