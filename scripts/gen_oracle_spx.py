import numpy as np
import matplotlib.pyplot as plt
import ray
import os
import pickle
import pathlib
import io
import re
import collections
import cv2
from skimage import segmentation
import scipy
import heapq
import math
import shapely
import shapely.ops

from PIL import Image
from PIL import ImageDraw
from skimage import measure
from matplotlib.colors import ListedColormap
# from gen_spx_visualization import draw_map, draw_map_overlay

_unknown_index, _known_index_start = 0, 1

def find_contours_of_connected_components(boundaries_map):

  def find_contours(binary):
    # RETR_EXTERNAL: find external contours only.
    # CHAIN_APPROX_NONE: do not approx.
    contours, _ = cv2.findContours(
      binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    return contours

  # we wanna assemble non-overlapping contours only.
  distinct_contours = []
  def add(contour):
    for other in distinct_contours:
      if contour.contains(other):
        return False
    distinct_contours.append(contour); return True

  for value in np.unique(boundaries_map):
    # for each pixel `value`, there could be multiple contours
    binary_contours_map = np.array(
      (boundaries_map == value), dtype=np.uint8)
    for contour in find_contours(binary_contours_map):
      # for some reason cv2.findContours() returns each contour
      # as a 3d numpy array with one redundant dimension
      num_coords, one, two = contour.shape
      if one != 1 or two != 2:
        raise AssertionError(
          "unexpected error: cv2 api might have been changed")
      # remove redundant (singleton) dimension
      contour = contour.squeeze(axis=1)
      if num_coords > 2:
        add(shapely.geometry.Polygon(contour))
      else:
        print("strange contour: %s"
          % list(tuple(xy) for xy in contour))

  return distinct_contours


def gt_connected_components_map(boundaries_map):

  # we just need the size of input image
  height, width = boundaries_map.shape
  # 32-bit signed interger pixels
  mode = "I"

  # placeholder with 'zero'-valued pixels as default
  components_map = Image.new(mode=mode, size=(width, height),
    color=_unknown_index)
  draw = ImageDraw.Draw(components_map, mode=mode)

  # find all distinct contours of connected components in
  # boundary-map.
  find_components = find_contours_of_connected_components
  distinct_contours = find_components(boundaries_map)

  # draw
  for index, contour in enumerate(distinct_contours,
      start=_known_index_start):
    draw.polygon(contour.exterior.coords, fill=index)

  return np.array(components_map, dtype=int)

  # we want all pixels to be assigned own colors
  # return segmentation.expand_labels(
  #   np.array(components_map, dtype=int), distance=100)


# ----------------------------------------------------------


import utils

def save_region_map(dir, map, cmap, image_name, image_path):
  # draw_map_overlay(
  #   Image.open(image_path), map, cmap, 0.5).save(
  #     dir / ("result/%s.png" % image_name))
  map = np.array(map, dtype=np.int16)
  with open(dir / ("label/%s.pkl" % image_name), "wb") as file:
    pickle.dump(
      {"labels": map, "valid_idxes": np.unique(map)}, file)

def main():

  from argparse import ArgumentParser
  from pathlib import Path

  parser = ArgumentParser()

  parser.add_argument("--dataset_name", required=True)
  parser.add_argument("--split", required=True)
  parser.add_argument("--outputs_dir", required=True, type=Path)

  conf = parser.parse_args()
  outputs_dir = conf.outputs_dir
  
  (outputs_dir / "label").mkdir(parents=True, exist_ok=True)
  (outputs_dir / "result").mkdir(parents=True, exist_ok=True)

  print("outputs saved to '%s'" % outputs_dir)

  # with open("distinct_rgb_colors_5000.pkl", "rb") as file:
  #   cmap = ListedColormap(pickle.load(file), N=5000)

  def _save_region_map(dir, map, image_name, image_path):
    save_region_map(dir, map, None, image_name, image_path)

  print("color map loaded")

  @utils.map_and_reduce
  def cityscapes_gt_cc(hash, image_path, true_label_path):
    print(hash)

    boundaries_map = Image.open(true_label_path).convert("L")
    boundaries_map = np.array(boundaries_map)

    components_map = gt_connected_components_map(boundaries_map)
    _save_region_map(outputs_dir, components_map, hash, image_path)

  if conf.dataset_name == 'pascal_voc_seg':
    devkit_path = './deeplab/datasets/pascal_voc_seg/'
    image_dir = devkit_path + 'VOC2012/JPEGImages'
    imageset_path = devkit_path + 'VOC2012/ImageSets/Segmentation/%s.txt'%conf.split
    semantic_segmentation_folder = devkit_path + 'VOC2012/SegmentationClassRaw'
    num_class = 21
  elif conf.dataset_name == 'cityscapes':
    devkit_path = './deeplab/datasets/cityscapes/'
    image_dir = devkit_path + 'leftImg8bit/'
    imageset_path = devkit_path + 'image_list/%s.txt'%conf.split
    semantic_segmentation_folder = devkit_path + 'gtFine'
    num_class = 19

  with open(imageset_path, 'r') as f:
    image_list = [line.strip() for line in f]

  image_hash_path = collections.OrderedDict()
  if conf.dataset_name == 'pascal_voc_seg':
    for image_name in image_list: 
      image_path = os.path.join(image_dir, image_name + '.jpg')
      true_label_path = Path(semantic_segmentation_folder) / (image_name + ".png")
      image_hash_path[image_name] = (image_path, true_label_path)
  elif conf.dataset_name == 'cityscapes':
    for image_name in image_list:
      parts = image_name.split("_")
      image_path = os.path.join(image_dir, conf.split, parts[0], image_name + '_leftImg8bit.png')
      true_label_path = Path(semantic_segmentation_folder) / conf.split / parts[0] / (image_name + "_gtFine_labelIds.png")
      image_hash_path[image_name] = (image_path, true_label_path)

  arg_tuples = []
  for n, (hash, (path, true_label_path)) in enumerate(image_hash_path.items()):
    arg_tuples.append((hash, path, true_label_path))
  for batch in utils.chunk(arg_tuples, 8):
    cityscapes_gt_cc(batch, 32)

if __name__ == "__main__":
  main()

