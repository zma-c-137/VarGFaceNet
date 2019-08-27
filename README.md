# VarGFaceNet

## License

The code of VarGFaceNet is released under the MIT License. There is no limitation for both acadmic and commercial usage.

## Introduction

This is a MXNET implementation of VarGFaceNet.
We achieved 1st place at Light-weight Face Recognition challenge/workshop on ICCV 2019(deepglint-light track)[LFR2019](https://ibug.doc.ic.ac.uk/resources/lightweight-face-recognition-challenge-workshop)

For details, please read the following papers:
* [VarGFaceNet: An Efficient Variable Group Convolutional Neural Network for Lightweight Face Recognition](???)
* [VarGNet: Variable Group Convolutional Neural Network for Efficient Embedded Computing](https://arxiv.org/abs/1907.05653)


## Base Module
![](https://github.com/zma-c-137/VarGFaceNet/blob/master/img/VarGFaceNet.png)

## Results

* train from scratch:

| Method  | LFW(%)  | CFP-FP(%) | AgeDB-30(%) | deepglint-light(%,TPR@FPR=1e-8) | 
| ------- | ------- | --------- | ----------- | ------------------------------- | 
|  Ours   | 0.99683 | 0.98086   | 0.98100     | 0.855                           |

* recursive knowledge distillation:

| Method      | LFW(%)  | CFP-FP(%) | AgeDB-30(%) | deepglint-light(%,TPR@FPR=1e-8) |
| ----------- | ------- | --------- | ----------- | ------------------------------- |
| recursive=1 | 0.99783 | 0.98400   | 0.98067     | 0.88334                         |
| recursive=2 | 0.99833 | 0.98271   | 0.98050     | 0.88784                         |

## Citation

If you find VarGFaceNet useful in your research, please consider to cite the following related papers:

```
@article{vargfacenet,
 author = {Yan, Mengjia and Zhao, Mengao and Xu, Zining and Zhang, Qian and Wang, Guoli and Su, Zhizhong},
 title = {VarGFaceNet: An Efficient Variable Group Convolutional Neural Network for Lightweight Face Recognition},
 journal = {In Proceedings of the IEEE International Conference on Computer Vision Workshops},
 year = 2019
}
@article{zhang2019vargnet,
  title={VarGNet: Variable Group Convolutional Neural Network for Efficient Embedded Computing},
  author={Zhang, Qian and Li, Jianjun and Yao, Meng and Song, Liangchen and Zhou, Helong and Li, Zhichao and Meng, Wenming and Zhang, Xuezhi and Wang, Guoli},
  journal={arXiv preprint arXiv:1907.05653},
  year={2019}
}
```

## Contact

```
[Mengao Zhao](mengao.zhao[at]gmail.com)
[Mengjia Yan](mengjyan[at]gmail.com)
```
