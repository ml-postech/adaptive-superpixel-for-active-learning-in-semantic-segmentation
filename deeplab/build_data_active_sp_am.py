# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts PASCAL VOC 2012 data to TFRecord file format with Example protos.

PASCAL VOC 2012 dataset is expected to have the following directory structure:

  + pascal_voc_seg
    - build_data.py
    - build_voc2012_data.py (current working directory).
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + ImageSets
          + Segmentation
    + tfrecord

Image folder:
  ./VOCdevkit/VOC2012/JPEGImages

Semantic segmentation annotations:
  ./VOCdevkit/VOC2012/SegmentationClass

list folder:
  ./VOCdevkit/VOC2012/ImageSets/Segmentation

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os.path
import sys
import datasets.build_data as build_data
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt
from kneed import DataGenerator, KneeLocator

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_name',
                           'pascal_voc_seg',
                           'Folder containing images.')
                           
tf.app.flags.DEFINE_string(
    'list_folder',
    './deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation',
    'Folder containing lists for training and validation')
    
tf.app.flags.DEFINE_string('image_folder',
                           './deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassRaw',
    'Folder containing semantic segmentation annotations.')
    
tf.app.flags.DEFINE_string(
    'tfrecord_dir',
    '',
    'Folder containing semantic segmentation annotations.')
    
tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder_region',
    '',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder_region_prev',
    '',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'region_idx_dir',
    '',
    'Folder containing semantic segmentation annotations.')
    
tf.app.flags.DEFINE_string(
    'valid_idx_dir_0',
    '',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'valid_idx_dir_1',
    '',
    'Folder containing semantic segmentation annotations.')
    
tf.app.flags.DEFINE_string(
    'train_split',
    'train',
    'Folder containing lists for training and validation')
    
tf.app.flags.DEFINE_integer('batch_id', 0, 'batch id')

tf.app.flags.DEFINE_string(
    'region_type',
    'rec',
    'region type: rec (rectangle) or sp (superpixel)') 

tf.app.flags.DEFINE_string(
    'is_uniq',
    'False',
    'whether assign a unique label to each region') 

tf.app.flags.DEFINE_float(
    'sieve_ratio',
    '0.5',
    'whether assign a unique label to each region') 

tf.app.flags.DEFINE_string(
    'job_name',
    'False',
    'whether assign a unique label to each region') 

tf.app.flags.DEFINE_string(
    'class_to_knee_dir',
    'False',
    'whether assign a unique label to each region')

tf.app.flags.DEFINE_integer('knee', 10, 'batch id')

_IGNORE_LABEL = 255
INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'ResizeBilinear_2:0'

def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  if FLAGS.dataset_name == 'pascal_voc_seg':
     image_reader = build_data.ImageReader('jpeg', channels=3)
     _NUM_SHARDS = 1
  elif FLAGS.dataset_name == 'cityscapes':
     image_reader = build_data.ImageReader('png', channels=3)
     _NUM_SHARDS = 10
     
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  image_list = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(image_list)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))  
  
  image_name_selected_regions = {}
  for i in range(FLAGS.batch_id + 1):
    dic = pickle.load(open(os.path.join(FLAGS.region_idx_dir, 'batch_{}.pkl'.format(i)), 'rb')) 
    for key in dic.keys():
        new_key = (key, i) # check !!
        image_name_selected_regions[new_key] = dic[key]
           
  prev_batch_id = FLAGS.batch_id - 1
  PATH_TO_FROZEN_GRAPH = os.path.join('outputs', FLAGS.job_name, 'batch_%d'%prev_batch_id, 'frozen_inference_graph.pb')
  label_reader = build_data.ImageReader('png', channels=1)

  seg_graph = tf.Graph()
  with seg_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
  
  correct_pixel, total_pixel = 0, 0
  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.tfrecord_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))

    if os.path.exists(output_filename):
      continue
    
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images) 
      
      with seg_graph.as_default():
        with tf.Session() as sess:
          for i in range(start_idx, end_idx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i + 1, len(image_list), shard_id))
            sys.stdout.flush()
            print(image_list[i])
            
            # Read the image.
            image_name = image_list[i]
            if FLAGS.dataset_name == 'pascal_voc_seg':              
               image_path = os.path.join(FLAGS.image_folder, image_name + '.jpg')   
               label_path = os.path.join(FLAGS.semantic_segmentation_folder, image_name + '.png')
            elif FLAGS.dataset_name == 'cityscapes':
               parts = image_name.split("_")
               image_path = os.path.join(FLAGS.image_folder, 'train', parts[0], image_name + '_leftImg8bit.png')
               label_path = os.path.join(FLAGS.semantic_segmentation_folder, FLAGS.train_split, parts[0], image_name + '_gtFine_labelTrainIds.png')
              
            new_image = Image.open(image_path)
            new_width, new_height = new_image.size 
            new_image = new_image.convert('RGB').resize((new_width//2, new_height//2))
            logits = sess.run( OUTPUT_TENSOR_NAME, feed_dict={INPUT_TENSOR_NAME : [np.asarray(new_image)]})
            logits = np.squeeze(logits)
            softmax_output = scipy.special.softmax(logits, axis=2) # (512, 1024, 19)

            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            height, width = image_reader.read_image_dims(image_data)
            seg_data = tf.gfile.FastGFile(label_path, 'rb').read()
            seg_height, seg_width = label_reader.read_image_dims(seg_data)
            seg_label = label_reader.decode_image(seg_data)
            seg_label_region = np.ones_like(seg_label, dtype = np.uint8) * 255

            for batch in range(FLAGS.batch_id + 1):
                selected_regions = image_name_selected_regions[(image_list[i], batch)]
                if batch == 0:
                    valid_idx_dir = FLAGS.valid_idx_dir_0
                else:
                    valid_idx_dir = FLAGS.valid_idx_dir_1 + "/batch_" + str(batch - 1) + "/label"
            
                if FLAGS.region_type == 'rec' :
                  rec_dic = pickle.load(open(os.path.join(FLAGS.valid_idx_dir, image_name + '.pkl'), 'rb'))
                  region_boxes = rec_dic['boxes']
                  
                  for j in selected_regions:
                    ymin = region_boxes[j, 0]
                    xmin = region_boxes[j, 1]
                    ymax = region_boxes[j, 2] 
                    xmax = region_boxes[j, 3] 
                                  
                    if FLAGS.is_uniq == 'False':
                        seg_label_region[ymin:ymax+1, xmin:xmax+1] = seg_label[ymin:ymax+1, xmin:xmax+1]
                    else: 
                        unique, unique_counts = np.unique(seg_label[ymin:ymax+1, xmin:xmax+1], return_counts=True)
                        dorm_class = unique[np.argmax(unique_counts)]
                        seg_label_region[ymin:ymax+1, xmin:xmax+1] = dorm_class

                elif FLAGS.region_type == 'sp':
                    sp_dic = pickle.load(open(os.path.join(valid_idx_dir, image_name + '.pkl'), 'rb'))
                    sp_labels = sp_dic['labels']
                    
                    for j in selected_regions:
                        y, x = np.where(sp_labels == j)
                        unique, unique_counts = np.unique(seg_label[y, x], return_counts=True)
                        dorm_label = unique[np.argmax(unique_counts)]

                        if dorm_label != 255:
                            confidence = softmax_output[(y/2).astype(int), (x/2).astype(int), dorm_label]
                            length = int(confidence.shape[0] / FLAGS.knee)
                            if length == 0 : length = 1
                            input_y = np.sort(confidence)[::length]
                            input_x = np.array([i for i in range(input_y.shape[0])])

                            if y.size >= 512: # merged
                                kneedle = KneeLocator(input_x, input_y, S=1, curve="concave", direction="increasing")
                                knee = kneedle.knee
                                if knee == None:
                                    knee = 0
                                sieve_value = input_y[knee]
                            else:
                                sieve_value = input_y[0]
                            
                            # kneedle.plot_knee()
                            # plt.show()
                            idx = np.where(softmax_output[(y/2).astype(int), (x/2).astype(int), dorm_label] >= sieve_value)[0]
                            seg_label_region[y[idx], x[idx]] = dorm_label

                            correct_pixel += idx.shape[0]
                            total_pixel += y.shape[0]
                            
            tmp_file = os.path.join(FLAGS.semantic_segmentation_folder_region, '{}.png'.format(image_name))
            pil_image = Image.fromarray(seg_label_region.squeeze())
            with tf.gfile.Open(tmp_file, mode='w') as f:
               pil_image.save(f, 'PNG')
            seg_data_region = tf.gfile.FastGFile(tmp_file, 'rb').read()
            
            # Convert to tf example.
            example = build_data.image_seg_to_tfexample(
                image_data, image_name, height, width, seg_data_region)
            tfrecord_writer.write(example.SerializeToString())
        
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):  
    
    dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, FLAGS.train_split + '.txt'))
       
    for dataset_split in dataset_splits:
       _convert_dataset(dataset_split)

if __name__ == '__main__':
  tf.app.run()


