# ST-SSL: Spatio-Temporal Self-Supervised Learning for Traffic Prediction 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-nycbike1)](https://paperswithcode.com/sota/traffic-prediction-on-nycbike1?p=spatio-temporal-self-supervised-learning-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-nycbike2)](https://paperswithcode.com/sota/traffic-prediction-on-nycbike2?p=spatio-temporal-self-supervised-learning-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-nyctaxi)](https://paperswithcode.com/sota/traffic-prediction-on-nyctaxi?p=spatio-temporal-self-supervised-learning-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-bjtaxi)](https://paperswithcode.com/sota/traffic-prediction-on-bjtaxi?p=spatio-temporal-self-supervised-learning-for)

This is a Pytorch implementation of ST-SSL in the following paper: 

* [J. Ji](https://echo-ji.github.io/academicpages/), J. Wang, C. Huang, et al. "[Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/25555)". in AAAI 2023.

![new](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **22/04/2022**: [The post of this paper](https://mp.weixin.qq.com/s/rMNsqYyfoeoysZxeVabh4w) is selected for a **headline** tweet by PaperWeekly and received nearly 7,000 reads. PaperWeekly is a leading AI academic platform in China.

## Requirement

We build this project by Python 3.8 with the following packages: 
```
numpy==1.21.2
pandas==1.3.5
PyYAML==6.0
torch==1.10.1
```

## Datasets

The datasets range from `{NYCBike1, NYCBike2, NYCTaxi, BJTaxi}`. Each dataset is composed of 4 files, namely `train.npz`, `val.npz`, `test.npz`, and `adj_mx.npz`.

```
|----NYCBike1\
|    |----train.npz
|    |----adj_mx.npz
|    |----test.npz
|    |----val.npz
```

train/val/test data is composed of 4 `numpy.ndarray` objects:

* `x`: a 4D tensor of shape (#timeslots, #lookback_window, #nodes, #flow_types)
* `y`: a 4D tensor of shape (#timeslots, #predict_horizon, #nodes, #flow_types). `x` and `y` are processed as a `sliding window view`.

* `x_offset`: a tensor indicating offsets of `x`'s lookback window. Note that the lookback window of data `x` is not consistent in time.
* `y_offset`: a tensor indicating offsets of `y`'s predict horizon.

For all datasets, previous 2-hour flows as well as previous 3-day flows around the predicted time are used to predict the flows for the next time step.

`adj_mx.npz` is a symmetric adjacency matrix, taking the value of 0 or 1.

⚠️ Note that all datasets are processed as a `sliding window view`. Raw data of **NYCBike1** and **BJTaxi** are collected from [STResNet](https://ojs.aaai.org/index.php/AAAI/article/view/10735). Raw data of **NYCBike2** and **NYCTaxi** are collected from [STDN](https://ojs.aaai.org/index.php/AAAI/article/view/4511).

## Model training and Evaluation

If the environment is ready, please run the following commands to train the model on the specific dataset from `{NYCBike1, NYCBike2, NYCTaxi, BJTaxi}`.
```bash
>> cd ST-SSL
>> ./runme 0 NYCBike1   # 0 gives the GPU id
```

This repo contains the NYCBike1 data. If you are interested in other datasets, please download them from [GitHub repo](https://github.com/Echo-Ji/ST-SSL_Dataset) or [Beihang Cloud Drive](https://bhpan.buaa.edu.cn/link/AAF30DD8F4A2D942F7A4992959335C2780).

## Cite

If you find the paper useful, please cite the following:

```
@article{ji2023modeling, 
  title={Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction}, 
  author={Ji, Jiahao and Wang, Jingyuan and Huang, Chao and Wu, Junjie and Xu, Boren and Wu, Zhenhe and Zhang Junbo and Zheng, Yu}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  volume={37},
  number={4},
  pages={4356-4364},
  year={2023}
}
```
