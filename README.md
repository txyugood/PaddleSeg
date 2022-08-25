# TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation（TransUNet 基于Paddle复现）
## 1.简介
医学图像分割是开发医疗保健系统的必要前提，尤其是疾病诊断和治疗规划。在各种医学图像分割任务中，u形结构（也称为UNet）已成为事实上的标准，并取得了巨大成功。然而，由于卷积运算的内在局部性，U-Net通常在显式建模长期依赖性时表现出局限性。设计用于seq2seq预测的Transformer已成为具有固有全局自我注意机制的替代架构，但由于低层次细节不足，可能导致有限的定位能力。在本文中，作者提出了Transunet作为医学图像分割的一种强有力的替代方法，它既有Transformer又有U-Net。一方面，Transformer将来自卷积神经网络（CNN）特征映射的标记化图像块编码为用于提取全局上下文的输入序列。另一方面，解码器对编码特征进行上采样，然后将其与高分辨率NN特征映射相结合，以实现精确定位。作者认为，Transformer可以作为医学图像分割任务的强编码器，与U-Net相结合，通过恢复局部空间信息来增强细节。TransUNet在不同的医学应用中，包括多器官分割和心脏分割，实现了优于各种竞争方法的性能。


## 2.复现精度
在Synapse数据集上的测试效果如下表。

| NetWork | epochs | opt  | batch_size | dataset | MDICE |
| --- | --- | --- | --- | --- | --- |
| TransUNet | 150 | SGD  | 24 | Synapse | 77.70% |

## 3.数据集
Synapse数据集下载地址:
使用作者提供的数据集，由于作者不允许分发。这里提供转换后的png图片数据。如有原数据需要可联系我。

[https://aistudio.baidu.com/aistudio/datasetdetail/165793](https://aistudio.baidu.com/aistudio/datasetdetail/165793)

其中R50+ViT-B_16.npz为预训练模型。


## 4.环境依赖
PaddlePaddle == 2.3.1
## 5.快速开始
### 训练：

下载数据集解压后，将数据集链接到项目的data目录下。

```shell
cd contrib/MedicalSeg/
mkdir data
cd data
ln -s path/to/Synapse_image Synapse_image
nohup python -u train.py --config configs/trans_unet/trans_unet_synapse.yml --do_eval --save_interval 1000 --has_dataset_json False --is_save_data False --num_workers 4 --log_iters 10  > train.log &
tail -f train.log
```

### 测试：

使用最优模型进行评估.

```shell
nohup python -u test.py --config configs/trans_unet/trans_unet_synapse.yml --model_path output/best_model/model.pdparams  --has_dataset_json False --is_save_data False > test.log &
tail -f test.log
```

config: 配置文件路径

model_path: 预训练模型路径

测试结果

```shell
W0825 13:56:06.053285 11744 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0825 13:56:06.053310 11744 gpu_resources.cc:91] device: 0, cuDNN Version: 7.6.
2022-08-25 13:56:08 [INFO]      Loading pretrained model from output/best_model/model.pdparams
2022-08-25 13:56:08 [INFO]      There are 400/400 variables loaded into VisionTransformer.
2022-08-25 13:56:08 [INFO]      Loaded trained params of model successfully
2022-08-25 13:56:09 [INFO]      Start evaluating (total_samples: 1568, total_iters: 1568)...
1568/1568 [==============================] - 104s 66ms/step - batch_cost: 0.0662 - reader cost: 3.5417e-04
2022-08-25 14:00:33 [INFO]      [EVAL] #Images: 1568, performance: 0.7770, mean_hd95: 8.523121
```


### 模型导出
模型导出可执行以下命令：

```shell
python export_model.py --config test_tipc/configs/transunet/trans_unet_synapse.yml --without_argmax --input_shape 1 1 1 224 224 --model_path=./output/best_model/model.pdparams --save_dir=./output/
```

参数说明：

config: 配置文件地址

model_path: 模型路径

save_dir: 模型保存路径

input_shape: 输入的数据形状

### Inference推理

可使用以下命令进行模型推理。该脚本依赖auto_log, 请参考下面TIPC部分先安装auto_log。infer命令运行如下：

```shell
python infer.py --use_swl False --use_warmup False --device=gpu --precision=fp32 --config=./output/deploy.yaml --batch_size=1 --image_path=./data/Synapse_dataset/train/images --benchmark=False --model_name=TransUNet
```

参数说明:


use_swl:是否使用swl

use_warmup: 是否热启动

device:使用设备GPU或CPU

batch_size: 批次大小

image_path: 输入图片路径

benchmark: 是否开启benchmark

precision: 运算精度

model_name: 模型名字



### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://gitee.com/Double_V/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```


```shell
bash test_tipc/prepare.sh test_tipc/configs/transunet/train_infer_python.txt "lite_train_lite_infer"

bash test_tipc/test_train_inference_python.sh test_tipc/configs/transunet/train_infer_python.txt "lite_train_lite_infer"
```

测试结果如截图所示：

<img src=./contrib/MedicalSeg/test_tipc/data/tipc_result.png></img>


## 6.代码结构与详细说明
```shell
MedicalSeg
├── configs         # 关于训练的配置，每个数据集的配置在一个文件夹中。基于数据和模型的配置都可以在这里修改
├── data            # 存储预处理前后的数据
├── deploy          # 部署相关的文档和脚本
├── medicalseg  
│   ├── core        # 训练和评估的代码
│   ├── datasets  
│   ├── models  
│   ├── transforms  # 在线变换的模块化代码
│   └── utils  
├── export.py
├── run-unet.sh     # 包含从训练到部署的脚本
├── tools           # 数据预处理文件夹，包含数据获取，预处理，以及数据集切分
├── train.py
├── val.py
└── visualize.ipynb # 用于进行 3D 可视化
```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| TransUNet |
|框架版本| PaddlePaddle==2.3.1|
|应用场景| 医疗图像分割 |
