# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:03:39 2019

@author: lile
"""
import os
import pickle
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('region_num_per_image', 1024, 'Number of regions per image.')
tf.app.flags.DEFINE_integer('batch_id', 0, 'Batch id')
tf.app.flags.DEFINE_integer('k', 1000, 'budget')
tf.app.flags.DEFINE_integer('seed', 1, 'selected region number for a batch')
tf.app.flags.DEFINE_integer('region_size', 32, 'selected region number for a batch')
tf.app.flags.DEFINE_string('feat_version', 'v0', 'the feature used for comparing region similarity')

tf.app.flags.DEFINE_string(
    'list_folder',
    '',
    'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string('region_uncert_dir',
                           '/home/lile/Projects/active_seg/region_uncertainty/voc2012/xception_41/baseline_slide_window_k_1000_train_iter_30000_bn_False_random_False_nms_0.6/batch_1',
                           'Folder containing images.')
                                                     
tf.app.flags.DEFINE_string('region_idx_dir',
                           '',
                           'selected region index')      
tf.app.flags.DEFINE_string(
    'train_split',
    'train',
    'train split')
                  
tf.app.flags.DEFINE_string('valid_idx_dir',
                           '/home/lile/Projects/active_seg/region_features/voc2012/resnet_v1_50',
                           'Folder containing images.') 

                        
tf.app.flags.DEFINE_string(
    'anno_cost_dir',
    'False',
    'Folder containing lists for training and validation') 
tf.app.flags.DEFINE_string(
    'cost_type',
    'rc',
    'cost type: rc (region count), cc (click count)') 

tf.app.flags.DEFINE_string('is_bal',
                           'False',
                           'Folder containing images.')    
 
tf.app.flags.DEFINE_string('class_to_region_idx_path',
                           '/home/lile/Projects/active_seg/region_features/voc2012/resnet_v1_50',
                           'Folder containing images.') 
   
def main(unused_argv):       
    imageset_path = FLAGS.list_folder + '/%s.txt' % (FLAGS.train_split)

    with open(imageset_path, 'r') as f:
        lines = f.readlines()
    image_list = [x.strip() for x in lines]

    region_num = FLAGS.region_num_per_image * len(image_list)
    print(region_num)
    
    def normalize(array, array_min, array_max):
        array = (array - array_min) / (array_max - array_min)
        return array
    
    selected_idx_prev = []
    if FLAGS.cost_type == 'cc':
        all_region_anno_cost = []
        for image_name in image_list:
            all_region_anno_cost.append(pickle.load(open(os.path.join(FLAGS.anno_cost_dir, image_name + '.pkl'), 'rb')))   
        all_region_anno_cost = np.hstack(all_region_anno_cost)
    
    # prepare uncertainty
    all_region_uncertainty_norm = np.zeros((region_num,1), dtype = np.float32) 
    
    all_region_uncertainty = []
    for image_name in image_list:
        all_region_uncertainty.append(pickle.load(open(os.path.join(FLAGS.region_uncert_dir, image_name + '.pkl'), 'rb')))   
    all_region_uncertainty = np.hstack(all_region_uncertainty)
    
    print('all_region_uncertainty max is {}, min is {}'.format(all_region_uncertainty.max(), all_region_uncertainty.min()))
    all_region_uncertainty_norm = normalize(all_region_uncertainty, all_region_uncertainty.min(), all_region_uncertainty.max())
    all_region_uncertainty_norm = all_region_uncertainty_norm.reshape(-1, 1)    
    all_region_uncertainty_norm[selected_idx_prev] = 0  
    
    if FLAGS.is_bal == 'True':
        class_to_region_idx = pickle.load(open(FLAGS.class_to_region_idx_path, 'rb')) 
        
        num_class = len(class_to_region_idx.keys())
        pixel_num_per_class = np.zeros((num_class,), dtype=np.int)
        total_pixel_num = 1024 * 2048 * 2975 # check !!
        for i in range(num_class):
            region_idx_and_size = np.array(class_to_region_idx[i])
            if region_idx_and_size.shape[0] == 0:
                pixel_num_per_class[i] = 0
            else:
                pixel_num_per_class[i] = np.sum(region_idx_and_size[:, 1])
        
        p = np.zeros((num_class,), dtype=np.float32)
        w = np.zeros((num_class,), dtype=np.float32)
        for i in range(num_class):
            p[i] = pixel_num_per_class[i] / total_pixel_num
            w[i]= np.exp(-p[i]) 

        region_to_w = np.ones((region_num,1), dtype=np.float32)
        for i in range(num_class):
            for reg, size in class_to_region_idx[i]: # check !!
                region_to_w[reg] = w[i]

    selected_idx = []
    cost = 0
         
    f = all_region_uncertainty_norm 
    if FLAGS.is_bal == 'True':
        f *= region_to_w
        
    sort_idx = np.argsort(f, axis=None)[::-1]
    
    if FLAGS.cost_type == 'rc':
        selected_idx = sort_idx[:FLAGS.k]
        print('selected last value is %f'%f[sort_idx[FLAGS.k-1]])
    else:
        p = 0
        while cost  < FLAGS.k:  
            sel_ind = sort_idx[p]
            assert sel_ind not in selected_idx
            assert sel_ind not in selected_idx_prev
            selected_idx.append(sel_ind)
            p +=1
            cost += all_region_anno_cost[sel_ind]
        print('selected last value is %f'%f[sel_ind])
    
    image_name_selected_regions = {}
    for image_name in image_list:
        image_name_selected_regions[image_name] = []
    
    selected_idx = np.array(selected_idx)
    print('selected %d region in batch %d'%(selected_idx.size, FLAGS.batch_id))
    for i in selected_idx:
        image_id = i // FLAGS.region_num_per_image
        region_id = i % FLAGS.region_num_per_image
    
        image_name = image_list[image_id]
        image_name_selected_regions[image_name].append(region_id)
    
    pickle.dump(image_name_selected_regions, open(os.path.join(FLAGS.region_idx_dir, 'batch_{}.pkl'.format(FLAGS.batch_id)), 'wb'))   
    pickle.dump(selected_idx, open(os.path.join(FLAGS.region_idx_dir, 'batch_{}_selected_idx.pkl'.format(FLAGS.batch_id)), 'wb')) 

if __name__ == '__main__':
  tf.app.run()
