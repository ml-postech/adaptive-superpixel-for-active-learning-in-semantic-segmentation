# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:33:14 2019

@author: lile
"""
import os
import pickle
import random
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('region_idx_dir',
                           '',
                           'Folder containing selected region index.')

tf.app.flags.DEFINE_string(
    'list_folder',
    '',
    'Folder containing lists for training and validation')
    
tf.app.flags.DEFINE_integer('region_num_per_image', 20, 'Number of regions per image.')
tf.app.flags.DEFINE_integer('batch_id', 0, 'batch id')
tf.app.flags.DEFINE_integer('seed', 10, 'seed for random generator')

tf.app.flags.DEFINE_integer('k', 5000, 'selected region number for a batch')
tf.app.flags.DEFINE_string(
    'train_split',
    'train',
    'split for training images') 
                           
tf.app.flags.DEFINE_string('valid_idx_dir',
                           '',
                           'valid region index for each image')  
                        
tf.app.flags.DEFINE_string(
    'anno_cost_dir',
    'False',
    'Folder containing annotation cost for each region') 

tf.app.flags.DEFINE_string(
    'cost_type',
    'rc',
    'cost type: rc (region count), cc (click count)') 

def random_select():
 
   imageset_path = FLAGS.list_folder + '/%s.txt' % (FLAGS.train_split)

   with open(imageset_path, 'r') as f:
       lines = f.readlines()
   image_list = [x.strip() for x in lines]

   image_name_selected_regions = {}
   for image_name in image_list:
       image_name_selected_regions[image_name] = []
   
   print('region_num_per_image is %d'%FLAGS.region_num_per_image)
   if FLAGS.batch_id == 0:
      region_idx = []
      n = 0
      for image_name in image_list:
          idxes = np.array(pickle.load(open(os.path.join(FLAGS.valid_idx_dir, image_name + '.pkl'), 'rb'))['valid_idxes'])
          idxes = idxes.astype(np.int32)
          idxes += n * FLAGS.region_num_per_image
          region_idx.append(idxes)
          n += 1
           
      random_idx = np.hstack(region_idx)       
      random.seed(FLAGS.seed)
      random.shuffle(random_idx)
      print('total region number is %d'%random_idx.size)
      pickle.dump(random_idx, open(os.path.join(FLAGS.region_idx_dir, 'random_idx.pkl'), 'wb'))
   else:
      random_idx = pickle.load(open(os.path.join(FLAGS.region_idx_dir, 'random_idx.pkl'), 'rb'))
   
   sel_k = FLAGS.k
   if FLAGS.batch_id == 0:   
        start_idx = 0
   else:
        start_idx = pickle.load(open(os.path.join(FLAGS.region_idx_dir, 'batch_{}_start_idx.pkl'.format(FLAGS.batch_id)), 'rb'))
   
   if FLAGS.cost_type == 'rc': 
           
        selected_idx = random_idx[start_idx: start_idx + sel_k]
        pickle.dump(start_idx + sel_k, open(os.path.join(FLAGS.region_idx_dir, 'batch_{}_start_idx.pkl'.format(FLAGS.batch_id+1)), 'wb'))
        
   elif FLAGS.cost_type == 'cc':
        all_region_anno_cost = []
        for image_name in image_list:
            all_region_anno_cost.append(pickle.load(open(os.path.join(FLAGS.anno_cost_dir, image_name + '.pkl'), 'rb')))   
        all_region_anno_cost = np.hstack(all_region_anno_cost)
        
        def get_selected_idx(start_idx, budget):
            selected_idx = []          
            cost = 0
            p = 0
            while cost < budget:
                cost += all_region_anno_cost[random_idx[start_idx+p]]
                selected_idx.append(random_idx[start_idx+p])
                p += 1
            return np.array(selected_idx), p       
                       
        selected_idx, p = get_selected_idx(start_idx, sel_k)
        pickle.dump(start_idx + p, open(os.path.join(FLAGS.region_idx_dir, 'batch_{}_start_idx.pkl'.format(FLAGS.batch_id+1)), 'wb'))
        
   print('selected %d region in batch %d'%(selected_idx.size, FLAGS.batch_id))
   for i in selected_idx:
     image_id = i // FLAGS.region_num_per_image
     region_id = i % FLAGS.region_num_per_image
     
     assert image_id in range(len(image_list))
     
     image_name = image_list[image_id] 
     image_name_selected_regions[image_name].append(region_id)
    
   pickle.dump(image_name_selected_regions, open(os.path.join(FLAGS.region_idx_dir, 'batch_{}.pkl'.format(FLAGS.batch_id)), 'wb'))
   pickle.dump(selected_idx, open(os.path.join(FLAGS.region_idx_dir, 'batch_{}_selected_idx.pkl'.format(FLAGS.batch_id)), 'wb'))     
    
def main(unused_argv):
    random_select()
  
if __name__ == '__main__':
    tf.app.run()
