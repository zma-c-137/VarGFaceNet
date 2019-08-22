# VarGFaceNet

## Introduction
This is a MXNET implementation of VarGFaceNet. We achieved 1st place at Light-weight Face Recognition challenge/workshop on ICCV 2019 [LFR2019](https://ibug.doc.ic.ac.uk/resources/lightweight-face-recognition-challenge-workshop)

Link to the original paper: ...


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
| ----------- | ------- | --------- | ----------- | ------------------------------- |
| recursive=2 | 0.99833 | 0.98271   | 0.98050     | 0.88784                         |
