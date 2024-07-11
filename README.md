<div align=center>
<h1> CardiacNet: Learning to Reconstruct Abnormalities for Cardiac Disease Assessment from Echocardiogram Videos </h1>
</div>
<div align=center>

   
<a src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square" href="https://xmengli.github.io/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square">
</a>

<a src="https://img.shields.io/badge/%F0%9F%9A%80-XiaoweiXu's Github-blue.svg?style=flat-square" href="https://github.com/XiaoweiXu/CardiacUDA-dataset">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-Xiaowei Xu's Github-blue.svg?style=flat-square">
</a>

</div>


## :hammer: PostScript
&ensp; :smile: This project is the pytorch implemention of **[[paper](https://arxiv.org/abs/2309.11145)]**.

&ensp; :laughing: Our experimental platform is configured with <u>Four *RTX3090 (cuda>=11.0)*</u>

&ensp; :smiley: The ***CardiacNet (PAH & ASD)*** are currently available at the : 
&ensp; &ensp; &ensp; &ensp; https://github.com/XiaoweiXu/CardiacNet-dataset.

## :computer: Installation


1. You need to build the relevant environment first, please refer to : [**requirements.yaml**](requirements.yaml)

2. Install Environment:
    ```
    conda env create -f requirements.yaml
    ```

+ We recommend you to use Anaconda to establish an independent virtual environment, and python > = 3.8.3; 


## :blue_book: Data Preparation

## *CardiacNet*
 1.  Please access the dataset through : [XiaoweiXu's Github](https://github.com/XiaoweiXu/)
 2.  Follw the instruction and download.
 3.  Finish dataset download and unzip the datasets.
 4.  Modify your code in the file:
        ```python
        Fine the file
        ..\train.py & ..\evaluate.py
        and 
        modify dataset path 
        in
        parser.add_argument('--dataset-path', type=str, default='your path', help='Path to data.') 
        or
        your can just use the command -- dataset_path='you path' when you train or evalute the model

## :feet: Training

1. In this framework, after the parameters are configured in the file **train.py**, you only need to use the command:

    ```python
    python train.py
2. You are also able to start distributed training. 

   - **Note:** Please set the number of graphics cards you need and their id in parameter **"enable_GPUs_id"**. For example, if you want to use the GPU with ID 3,4,5,6 for training, just enter the 3,4,5,6 in args.enable_GPUs_id.

## :feet: Evaluation

1. For the evaluation, you can directly use the following command:

    ```python
    python evaluate.py
    
    Note that:
    Before you evaluate the model, remember to modified the saved model path in the file evaluate.py, 
    which is args.checkpoint_path


###### :rocket: Code Reference 
  - https://github.com/CompVis/taming-transformers

###### :rocket: Updates Ver 1.0（PyTorch）
###### :rocket: Project Created by Jiewen Yang : jyangcu@connect.ust.hk