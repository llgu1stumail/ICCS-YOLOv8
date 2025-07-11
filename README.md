# ICCS-YOLOv8
 
# Title 

Enhanced YOLOv8 with Global Attention for Precise Continuous Casting Slab Detection

# Abstract 

Continuous casting slabs are vital intermediates in steel production, necessitating real-time and accurate detection. Traditional methods face challenges such as sensor mis-triggering and difficulty in detecting small targets. This paper introduces ICCS-YOLOv8, an enhanced YOLOv8 algorithm incorporating a Partial Transformer Block for improved small target feature extraction, Selective Boundary Aggregation for handling size variations and occlusions, and a dedicated small target detection layer. Experimental results on a self-constructed dataset demonstrate a 97.4% mean average precision (mAP)@50 with a computational complexity of 13.1 GFLOPs, achieving a balance between accuracy and efficiency. The model meets real-time detection requirements (≥15 FPS) in industrial settings.

# Datasets

1.self-constructed dataset: https://github.com/llgu1stumail/11ICCS-YOLOv8
2.Visdrone dataset:   https://github.com/VisDrone/VisDrone-Dataset


# dependency installation

    1. 执行pip uninstall ultralytics把安装在环境里面的ultralytics库卸载干净.<这里需要注意,如果你也在使用yolov8,最好使用anaconda创建一个虚拟环境供本代码使用,避免环境冲突导致一些奇怪的问题>
    2. 卸载完成后同样再执行一次,如果出现WARNING: Skipping ultralytics as it is not installed.证明已经卸载干净.
    3. 如果需要使用官方的CLI运行方式或者多卡运行,需要把ultralytics库安装一下,执行命令:<pip install -e .>,请注意此命令需要在本项目的路径下执行,当然安装后对本代码进行修改依然有效.注意:不需要使用(官方的CLI运行方式、多卡运行),可以选择跳过这步.
    4. 额外需要的包安装命令:
        pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
        以下主要是使用dyhead必定需要安装的包,如果安装不成功dyhead没办法正常使用!如果执行了还是不成功,可看最下方mmcv安装问题.
        pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple
        mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
        mim install "mmcv>=2.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
    5. 运行时候如果还缺什么包就请自行安装即可.(For more details,please refer to: requirements.txt)


# 自带的一些文件说明

   1. train.py
       训练模型的脚本（生成Table 3-10的所有数据、Fig. 9、Fig.10、Fig.12）
   2. main_profile.py
       输出模型和模型每一层的参数,计算量的脚本
   3. val.py
       使用训练好的模型计算指标的脚本
   4. detect.py
       推理的脚本
   5. test_yaml.py
       用来测试所有yaml是否能正常运行的脚本
   6. heatmap.py  （生成Fig. 11）
       生成热力图的脚本
   7. get_FPS.py（生成Table 9中的FPS参数）
       计算模型储存大小、模型推理时间、FPS的脚本

# 模型配置文件

模型配置文件都在ultralytics/cfg/models/v8中.
yolov8有五种大小的模型,以下模型参数量和计算量均为类别80且重参数化后计算.

    YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
    YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
    YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
    YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
    YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs


# 注意⚠️ 所有路经均要替换成文件下载之后的对应路径才能正确运行文件

