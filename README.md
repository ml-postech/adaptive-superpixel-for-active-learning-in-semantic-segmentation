# Adaptive Superpixel for Active Learning in Semantic Segmentation
This repository is the official implementation of ["Adaptive Superpixel for Active Learning in Semantic Segmentation"](https://arxiv.org/abs/2303.16817) accepted by ICCV 2023.

## Abstract
Learning semantic segmentation requires pixel-wise annotations, which can be time-consuming and expensive. To reduce the annotation cost, we propose a superpixel-based active learning (AL) framework, which collects a dominant label per superpixel instead. To be specific, it consists of adaptive superpixel and sieving mechanisms, fully dedicated to AL. At each round of AL, we adaptively merge neighboring pixels of similar learned features into superpixels. We then query a selected subset of these superpixels using an acquisition function assuming no uniform superpixel size. This approach is more efficient than existing methods, which rely only on innate features such as RGB color and assume uniform superpixel sizes. Obtaining a dominant label per superpixel drastically reduces annotators' burden as it requires fewer clicks. However, it inevitably introduces noisy annotations due to mismatches between superpixel and ground truth segmentation. To address this issue, we further devise a sieving mechanism that identifies and excludes potentially noisy annotations from learning. Our experiments on both Cityscapes and PASCAL VOC datasets demonstrate the efficacy of adaptive superpixel and sieving mechanisms.

## Usages
Our code is written based on ["Revisiting Superpixels for Active Learning in Semantic Segmentation With Realistic Annotation Costs"](https://github.com/cailile/Revisiting-Superpixels-for-Active-Learning).
You first prepare Cityscapes dataset and Xception-65 model pretrained on ImageNet. For the Cityscapes dataset, you can refer to [DeepLab on Cityscapes](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md). For the Xception-65 model, you can refer to [DeepLab Model Zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md). Final directory structure is depicted as:

```
+ datasets
  + cityscapes
    + leftImg8bit
    + gtFine
    + gtFineRegion
    + image_list
    + tfrecord

+ models
  + xception_65
```

To obtain base and oracle superpixels, you run python ./scripts/extract_superpixels.py and python ./scripts/gen_oracle_spx.py, respectively. For warm-up round, you run bash ./bash_files/warm_up.sh. After the warm-up, you can generate adaptive superpixels for the next round using python ./scripts/gen_adaptive_spx.py and a previously trained model. You then run bash ./bash_files/adaptive_superpixel.sh for subsequent rounds.

## Cite
Please cite our paper if you use the model or this code in your own work:
```
@inproceedings{kim2023adaptive,
  title={Adaptive Superpixel for Active Learning in Semantic Segmentation},
  author={Hoyoung Kim and Minhyeon Oh and Sehyun Hwang and Suha Kwak and Jungseul Ok},
  booktitle=ICCV,
  year={2023},
  url={https://arxiv.org/abs/2303.16817}
}
```
