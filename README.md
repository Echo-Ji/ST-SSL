# ST-SSL: Spatio-Temporal Self-Supervised Learning for Traffic Prediction 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-nycbike1)](https://paperswithcode.com/sota/traffic-prediction-on-nycbike1?p=spatio-temporal-self-supervised-learning-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-nycbike2)](https://paperswithcode.com/sota/traffic-prediction-on-nycbike2?p=spatio-temporal-self-supervised-learning-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-nyctaxi)](https://paperswithcode.com/sota/traffic-prediction-on-nyctaxi?p=spatio-temporal-self-supervised-learning-for)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spatio-temporal-self-supervised-learning-for/traffic-prediction-on-bjtaxi)](https://paperswithcode.com/sota/traffic-prediction-on-bjtaxi?p=spatio-temporal-self-supervised-learning-for)

This is a Pytorch implementation of ST-SSL in the following paper: 

* [J. Ji](https://echo-ji.github.io/academicpages/), J. Wang, C. Huang, et al. "[Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/25555)". in AAAI 2023.

![framework](https://github.com/Echo-Ji/ST-SSL/assets/23469289/57d66470-5e12-4f27-9558-21cbb34b3a64)

![new](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **27/10/2023**: This paper is picked up by leading WeChat official accounts in the field of data mining and transportation. [当交通遇上机器学习](https://mp.weixin.qq.com/s/eI26ORLsJe_20WMpA5UeKA) | [时空实验室](https://mp.weixin.qq.com/s/CBKkyeSBTXOya2Cg3sgj7g) | [AI蜗牛车](https://mp.weixin.qq.com/s/vbczwY0UmzF7nBawEHpuaQ)

![new](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **22/04/2023**: [The post of this paper](https://mp.weixin.qq.com/s/rMNsqYyfoeoysZxeVabh4w) is selected for a **headline** tweet by PaperWeekly and received nearly 7,000 reads. PaperWeekly is a leading AI academic platform in China.

![new](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **09/02/2023**: The [video replay](https://underline.io/events/380/posters/14098/poster/68914-584-spatio-temporal-self-supervised-learning-for-traffic-flow-prediction) of academic presentation at AAAI 2023.


![new](https://github.com/RUCAIBox/RecBole/blob/master/asset/new.gif) **04/02/2023**: J. Ji is invited to give a talk at AAAI 2023 Beijing Pre-Conference. The talk is about [Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction](https://event.baai.ac.cn/activities/650).
## Requirement

We build this project by Python 3.8 with the following packages: 
```
numpy==1.21.2
pandas==1.3.5
PyYAML==6.0
torch==1.10.1
```

## Datasets

The datasets range from `{NYCBike1, NYCBike2, NYCTaxi, BJTaxi}`. You can download them from [GitHub repo](https://github.com/Echo-Ji/ST-SSL_Dataset), [Beihang Cloud Drive](https://bhpan.buaa.edu.cn/link/AAF30DD8F4A2D942F7A4992959335C2780), or [Google Drive](https://drive.google.com/file/d/1n0y6X8pWNVwHxtFUuY8WsTYZHwBe9GeS/view?usp=sharing).

Each dataset is composed of 4 files, namely `train.npz`, `val.npz`, `test.npz`, and `adj_mx.npz`.

```
|----NYCBike1\
|    |----train.npz
|    |----adj_mx.npz
|    |----test.npz
|    |----val.npz
```

train/val/test data is composed of 4 `numpy.ndarray` objects:

* `x`: a 4D tensor of shape `(#timeslots, #lookback_window, #nodes, #flow_types)`
* `y`: a 4D tensor of shape `(#timeslots, #predict_horizon, #nodes, #flow_types)`. `x` and `y` are processed as a `sliding window view`.

* `x_offset`: a tensor indicating offsets of `x`'s lookback window. Note that the lookback window of data `x` is not consistent in time.
* `y_offset`: a tensor indicating offsets of `y`'s predict horizon.

For all datasets, previous 2-hour flows as well as previous 3-day flows around the predicted time are used to predict the flows for the next time step.

`adj_mx.npz` is a symmetric adjacency matrix, taking the value of 0 or 1.

⚠️ Note that all datasets are processed as a sliding window view. Raw data of **NYCBike1** and **BJTaxi** are collected from [STResNet](https://ojs.aaai.org/index.php/AAAI/article/view/10735). Raw data of **NYCBike2** and **NYCTaxi** are collected from [STDN](https://ojs.aaai.org/index.php/AAAI/article/view/4511).

## Model training and Evaluation

If the environment is ready, please run the following commands to train the model on the specific dataset from `{NYCBike1, NYCBike2, NYCTaxi, BJTaxi}`.
```bash
>> cd ST-SSL
>> ./runme 0 NYCBike1   # 0 specifies the GPU id
```

Note that this repo only contains the NYCBike1 data because including all datasets can make this repo heavy.

## Cite

If you find the paper useful, please cite the following:

```
@article{ji2023spatio, 
  title={Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction}, 
  author={Ji, Jiahao and Wang, Jingyuan and Huang, Chao and Wu, Junjie and Xu, Boren and Wu, Zhenhe and Zhang Junbo and Zheng, Yu}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  volume={37},
  number={4},
  pages={4356-4364},
  year={2023}
}
```
