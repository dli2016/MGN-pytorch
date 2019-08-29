# Multiple Granularity Network
Reproduction of paper:[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)
## Acknowledgements
- [MGN-pytorch](https://github.com/seathiefwang/MGN-pytorch)
- [person-reid-triplet-loss-baseline](https://github.com/huanghoujing/person-reid-triplet-loss-baseline)
- [RAP](https://github.com/dangweili/RAP)
- [Market1501](http://www.liangzheng.org/Project/project_reid.html)

## Differences
- The training datasets should be orgnized with the format of [person-reid-triplet-loss-baseline](https://github.com/huanghoujing/person-reid-triplet-loss-baseline)
- All the datasets (e.g., rap, market1501, and so on) met the above format can be utilized to train a MGN model.

## Dependencies

- Python >= 3.5
- PyTorch >= 0.4.0
- TorchVision
- Matplotlib
- Argparse
- Sklearn
- Pillow
- Numpy
- Scipy
- Tqdm

## Train

### Prepare training data

Download Market1501 training data.[here](http://www.liangzheng.org/Project/project_reid.html)

### Begin to train

In the demo.sh file, add the dataset directory to --datadir

run `sh demo.sh`

##  Result (without rerank)

| Dateset | mAP | rank1 | rank3 | rank5 | rank10 |
| :------: | :------: | :------: | :------: | :------: | :------: |
| market1501 | 87.55 | 94.69 | 97.54 | 98.19 | 98.87 |
| rap | 66.69 | 84.66 | 90.35 | 92.38 | 94.45 |

Download model file (fine-tuned on market1501) in [here](https://pan.baidu.com/s/1DbZsT16yIITTkmjRW1ifWQ)

Download model file (fine-tuned on RAP) in [here](https://pan.baidu.com/s/1hOlJWi1pfiB-_LNjki0uug), password: ysx8 .

## The architecture of Multiple Granularity Network (MGN)
![Multiple Granularity Network](https://pic2.zhimg.com/80/v2-90a8763a0b7aa86d9152492eb3f85899_hd.jpg)

Figure . Multiple Granularity Network architecture.

```text
@ARTICLE{2018arXiv180401438W,
    author = {{Wang}, G. and {Yuan}, Y. and {Chen}, X. and {Li}, J. and {Zhou}, X.},
    title = "{Learning Discriminative Features with Multiple Granularities for Person Re-Identification}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1804.01438},
    primaryClass = "cs.CV",
    keywords = {Computer Science - Computer Vision and Pattern Recognition},
    year = 2018,
    month = apr,
    adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180401438W},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
