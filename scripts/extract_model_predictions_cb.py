#%%
import pickle
import os
import math
from PIL import Image
import scipy.ndimage
import numpy as np
import tensorflow as tf
import scipy.special

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_name', 'cityscapes', 'semantic segmentation dataset name')
tf.app.flags.DEFINE_string('job_name', 'cityscapes_xception_65_train_iter_60000_bn_True_trainbs_4_crop_769_nr_2148_rt_sp_seeds_ct_rc_uniq_True_v0_run_1', 'name for the experiment')
tf.app.flags.DEFINE_integer('region_num_per_image', 2048, 'Number of regions per image.')
tf.app.flags.DEFINE_integer('num_superpixels', 2048, 'Number of superpixels per image.')
tf.app.flags.DEFINE_integer('region_size', 32, 'region size')
tf.app.flags.DEFINE_integer('batch_id', 0, 'batch id')
tf.app.flags.DEFINE_string('region_type', 'sp', 'rec or sp')
tf.app.flags.DEFINE_string('sp_method', 'seeds', 'method for superpixel generation')
tf.app.flags.DEFINE_string('split', 'train', 'the feature used for comparing region similarity')
tf.app.flags.DEFINE_string('is_bal',
                           'True',
                           'whether use class-balanced sampling in active selection')
tf.app.flags.DEFINE_string('is_impurity',
                           'False',
                           'whether use impurity in active selection')
tf.app.flags.DEFINE_float('alpha', 0.5, 'ratio of impurity') 
tf.app.flags.DEFINE_string('superpixel_label_dir', '', '')

INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'ResizeBilinear_2:0'


def compute_sp_pseudo_label_uncertainty(image_list, image_name_path, num_class):
    
    num_superpixels = FLAGS.num_superpixels
    region_num_per_image = FLAGS.region_num_per_image
    superpixel_label_dir = FLAGS.superpixel_label_dir

    if FLAGS.is_bal == 'True':
        class_to_region_idx_dir = './class_to_region_idx/%s/batch_%d'%(FLAGS.job_name, FLAGS.batch_id)
        if not os.path.exists(class_to_region_idx_dir):
            os.makedirs(class_to_region_idx_dir)
    
    region_uncert_dir = './region_uncertainty/%s/batch_%d'%(FLAGS.job_name, FLAGS.batch_id)
    
    PATH_TO_FROZEN_GRAPH = os.path.join('outputs', FLAGS.job_name, 'batch_%d'%FLAGS.batch_id, 'frozen_inference_graph.pb')

    seg_graph = tf.Graph()
    with seg_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    n = 0                
    with seg_graph.as_default():
      with tf.Session() as sess:  
        for image_name in image_list:
            if os.path.exists(os.path.join(region_uncert_dir, image_name + '.pkl')):
                if FLAGS.is_bal == 'True':
                    if os.path.exists(os.path.join(class_to_region_idx_dir + '/' + image_name + '.pkl')):
                        continue
                else:
                    continue
            image_path = image_name_path[image_name]
            print(image_name)
            
            image = Image.open(image_path)
            width, height = image.size
            if FLAGS.dataset_name == 'cityscapes':
                resize_factor = 2
                image = image.convert('RGB').resize((width//resize_factor, height//resize_factor))
            
            logits = sess.run( OUTPUT_TENSOR_NAME, feed_dict={INPUT_TENSOR_NAME: [np.asarray(image)]})
            logits = np.squeeze(logits)
            
            softmax_output = scipy.special.softmax(logits, axis = 2)   
            softmax_output_sorted = np.sort(softmax_output, axis = 2)
            uncertainty_output = softmax_output_sorted[:, :, -2] / softmax_output_sorted[:, :, -1]
            pred = np.argmax(logits, axis = 2)
                
            uncertainty_array = np.zeros((FLAGS.region_num_per_image, ))
            
            if FLAGS.is_bal == 'True':
                class_to_region_idx = {}
                for i in range(num_class):
                    class_to_region_idx[i] = []
                    
            sp_dic = pickle.load(open(os.path.join(superpixel_label_dir, image_name + '.pkl'), 'rb'))
            labels = sp_dic['labels']
            if labels.shape[0] > logits.shape[0]:
                labels = scipy.ndimage.zoom(labels, (0.5, 0.5), order = 0)
            
            for j in sp_dic['valid_idxes']:
                y, x = np.where(labels == j)
                    
                if y.size > 0:
                    if FLAGS.is_bal == 'True':
                        unique, unique_counts = np.unique(pred[y, x], return_counts=True)
                        class_id = unique[np.argmax(unique_counts)]
                        class_to_region_idx[class_id].append([j + n * region_num_per_image, y.size]) # check

                    uncertainty_array[j] = np.mean(uncertainty_output[y, x])
            
            assert np.isnan(np.sum(uncertainty_array)) == False  
                        
            pickle.dump(uncertainty_array, open(os.path.join(region_uncert_dir, image_name + '.pkl'), 'wb'))         
            if FLAGS.is_bal == 'True':
                pickle.dump(class_to_region_idx, open(class_to_region_idx_dir + '/' + image_name + '.pkl', 'wb'))
            n += 1
     
                                     
def compute_rec_pseudo_label_uncertainty(image_list, image_name_path, num_class):
    
    valid_idx_dir = os.path.join('rectangles', FLAGS.dataset_name, 'rs_%d'%FLAGS.region_size)
    
    class_to_region_idx_dir = './class_to_region_idx/%s/batch_%d'%(FLAGS.job_name, FLAGS.batch_id)
    if not os.path.exists(class_to_region_idx_dir):
            os.makedirs(class_to_region_idx_dir)
    
    region_uncert_dir = './region_uncertainty/%s/batch_%d'%(FLAGS.job_name, FLAGS.batch_id)        
    PATH_TO_FROZEN_GRAPH = os.path.join('outputs', FLAGS.job_name, 'batch_%d'%FLAGS.batch_id, 'frozen_inference_graph.pb')

    seg_graph = tf.Graph()
    with seg_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    n = 0         
    with seg_graph.as_default():
      with tf.Session() as sess:  
        for image_name in image_list:
            if os.path.exists(os.path.join(region_uncert_dir, image_name + '.pkl')):
                continue
            image_path = image_name_path[image_name]
            print(image_name)
            
            image = Image.open(image_path)
            width, height = image.size
            if FLAGS.dataset_name == 'cityscapes':
                resize_factor = 2
                image = image.convert('RGB').resize((width//resize_factor, height//resize_factor))
            
            logits = sess.run( OUTPUT_TENSOR_NAME, feed_dict={INPUT_TENSOR_NAME: [np.asarray(image)]})
            logits = np.squeeze(logits)
             
            softmax_output = scipy.special.softmax(np.squeeze(logits), axis = 2)
            softmax_output_sorted = np.sort(softmax_output, axis = 2)
            uncertainty_output = softmax_output_sorted[:, :, -2] / softmax_output_sorted[:, :, -1]
            pred = np.argmax(logits, axis = 2)
                    
            if FLAGS.dataset_name == 'cityscapes':
                pred = scipy.ndimage.zoom( pred, (2, 2), order = 0)
                uncertainty_output = scipy.ndimage.zoom(uncertainty_output, (2, 2), order = 1)
            
            rec_dic = pickle.load(open(os.path.join(valid_idx_dir, image_name + '.pkl'), 'rb'))
            region_boxes = rec_dic['boxes']
            valid_idxes = rec_dic['valid_idxes']
            
            if FLAGS.is_bal == 'True':
                class_to_region_idx = {}
                for i in range(num_class):
                    class_to_region_idx[i] = []
            uncertainty_array = np.zeros((FLAGS.region_num_per_image, ))
            
            for j in valid_idxes:
                ymin = region_boxes[j, 0]
                xmin = region_boxes[j, 1]
                ymax = region_boxes[j, 2] 
                xmax = region_boxes[j, 3]
                
                if FLAGS.is_bal == 'True':
                    unique, unique_counts = np.unique(pred[ymin:ymax+1, xmin:xmax+1], return_counts=True)
                    class_id = unique[np.argmax(unique_counts)]
                    class_to_region_idx[class_id].append(j+ n * FLAGS.region_num_per_image)
    
                uncertainty_array[j] = np.mean(uncertainty_output[ymin:ymax+1, xmin:xmax+1])
            
            assert np.isnan(np.sum(uncertainty_array)) == False
            
            pickle.dump(uncertainty_array, open(os.path.join(region_uncert_dir, image_name + '.pkl'), 'wb'))        
            if FLAGS.is_bal == 'True':
                pickle.dump(class_to_region_idx, open(class_to_region_idx_dir + '/' + image_name + '.pkl', 'wb')) 
            n += 1 
                                
def combine_class_to_region_idx(image_list, num_class, ignore_label):
        
        class_to_region_idx_dir = './class_to_region_idx/%s/batch_%d'%(FLAGS.job_name, FLAGS.batch_id)
        
        class_to_region_idx_path = class_to_region_idx_dir + '/ctr_idx.pkl'
        
        if  os.path.exists(class_to_region_idx_path):
            return
                
        class_to_region_idx_all = {}
        for i in range(num_class):
            class_to_region_idx_all[i] = []

        for image_name in image_list:
            class_to_region_idx = pickle.load(open(class_to_region_idx_dir + '/' + image_name + '.pkl', 'rb'))
            for c in class_to_region_idx:
                if c != ignore_label:
                   class_to_region_idx_all[c].extend(class_to_region_idx[c])   
        
        pickle.dump(class_to_region_idx_all, open(class_to_region_idx_path, 'wb')) 
                        
if __name__ == '__main__':
    if FLAGS.dataset_name == 'pascal_voc_seg':
            devkit_path = './deeplab/datasets/pascal_voc_seg/VOCdevkit/'
            image_dir = devkit_path + 'VOC2012/JPEGImages'
            imageset_path = devkit_path + 'VOC2012/ImageSets/Segmentation/%s.txt'%FLAGS.split
            semantic_segmentation_folder = devkit_path + 'VOC2012/SegmentationClassRaw'
            num_class = 21
    elif FLAGS.dataset_name == 'cityscapes':
            devkit_path = './deeplab/datasets/cityscapes/'
            image_dir = devkit_path + 'leftImg8bit/'
            imageset_path = devkit_path + 'image_list/%s.txt'%FLAGS.split
            semantic_segmentation_folder = devkit_path + 'gtFine'
            num_class = 19

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
        
    ignore_label = 255
    
    if FLAGS.region_type == 'sp':
        if FLAGS.is_impurity == 'True':
            compute_sp_pseudo_label_uncertainty_impurity(image_list, image_name_path, num_class)
        else:
            compute_sp_pseudo_label_uncertainty(image_list, image_name_path, num_class)
    elif FLAGS.region_type == 'rec':
        compute_rec_pseudo_label_uncertainty(image_list, image_name_path, num_class)
    
    if FLAGS.is_bal == 'True':       
       combine_class_to_region_idx(image_list, num_class, ignore_label)


           
              