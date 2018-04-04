# Scene Understanding for Autonomous Vehicles

### Master in Computer Vision - M5 Visual recognition

## Team "Avengers"
- Mikel Menta Grade - mikel.menta@e-campus.uab.cat
- Alex Vallès Fernández - alex.valles@e-campus.uab.cat
- Sebastian Maya Hernández - sebastiancamilo.maya@e-campus.uab.cat
- Pedro Luis Trigueros Mondéjar - pedroluis.trigueros@e-campus.uab.cat

### Project overview
In this project we focus on scene understanding for autonomous vehicles. Understanding the context of the own vehicle is key for autonomous driving. The project consists of three parts or stages, corresponding to object detection, recognition/classification and semantic segmentation.<br/>
Furthermore, we aim to learn the basic concepts, techniques, tricks and libraries to develop and evaluate deep neural networks.



### Documentation
- [Overleaf article](https://www.overleaf.com/read/fbhxbjqydfwx)
- [Google Slides](https://docs.google.com/presentation/d/1ciF9LLclwjYReKp43KqC6W29ly3jamvE6WkJK1HoqTQ/edit?usp=sharing)

### Paper Summaries
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://www.overleaf.com/read/jwgpqfzvnwgk)
- [Squeeze and Excitation Blocks](https://www.overleaf.com/read/bxmpwjqhckmn)
- [You Only Look Once (YOLO)](https://www.overleaf.com/read/fvtfvvwbptsv)
- [Focal Loss for Dense Object Detection](https://www.overleaf.com/read/xsknbrtmsbgd)
- [Fully Convolutional Networks for Semantic Segmentation](https://drive.google.com/file/d/1AfJxdUBDgSleze1uJbtyS479wYe7Q6Pj/view?usp=sharing)
- [Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://drive.google.com/file/d/1nnYB8hGM77_PQLIwVtui7OVhnv4Uy17a/view?usp=sharing)

### Weights of the models
The weights, experiments' info and the TensorBoard logs are available [here](https://drive.google.com/drive/folders/15mccIEOhimkFo_5K3M43Vmu9xT0CEvpC?usp=sharing).

### Instructions for use code 
```
python train.py -c config/configFile.py -e expName
```

## Object Recognition (Week 2)

<p align="justify"><b>Overview:</b>
We have implemented the winner architecture of ILSVRC 2017 Classification Competition, the Squeeze-Excitation ResNet. 
The squeeze-excitation block consists on an squeeze step based on a Global Average Pooling over the output of the residual block and afterwards, an excitation step based on obtaining some weights for each output channel of the residual block and multiplying the channels by those weights. To obtain those weights, two fully-connected layers are used. The first one performs a dimensionality reduction over the number of channels C and uses ReLU activation. The reduction has been performed with a reduction ratio of r=16. The second FC layer  recovers the original dimensionality C and uses sigmoid activation for obtaining a weight in range [0,1].
</p>

<p align="justify"><b>Datasets:</b>

[TT100K](http://cg.cs.tsinghua.edu.cn/traffic-sign/)<br/>
[BelgiumTSC](http://btsd.ethz.ch/shareddata/)<br/>
[KITTI](http://www.cvlibs.net/datasets/kitti/)<br/>

</p>

<b>Contributions to the code:</b>
+ `code/models/se_resnet50.py` - Squeeze and Excitation ResNet implementation.  
+ `code/scripts/dataset_analysis.py` - Script for analysing the datasets.
+ `code/config/*` - Configuration files for image classification 

<b>Completeness of the goals:</b>

- [x] a) Run the provided code.<br> 
		- Analyze the dataset.<br> 
		- Calculate the accuracy on train and test sets.<br> 
		- Evaluate different techniques in the configuration file.<br> 
		- Transfer learning to another dataset (BTS).<br> 
		- Understand which parts of the code are doing what you specify in the configuration file.<br> 
- [x] b) Train a network on another dataset.<br> 
- [x] c) Implement a new network (c.2 - Develop the network entirely by yourself).<br> 
- [x] e) Report showing the achieved results.<br>

<p align="justify"><b>Results:</b>

| Neuronal Network          | Dataset     | Accuracy training  | Accuracy test |
| ------------------------- |:-----------:|:------------------:|:-------------:|
|VGG                        | TT100K      | 0.9664             | 0.9546        |
|VGG                        | BelgiumTSC  | 0.9875             | 0.9607        |
|VGG                        | KITTI       | 0.7950             | 0.7805        |
|Squeeze-Excitation ResNet  | TT100K      | 0.9987             | 0.9619        |
|Squeeze-Excitation ResNet  | BelgiumTSC  | 0.9978             | 0.9655        |
|VGG with Crop (224,224)    | TT100K      | 0.9513             | 0.9226        |
|VGG pretrained on ImageNet | TT100K      | 0.6610             | 0.7859        |
|VGG pretrained on TT100K   | BelgiumTSC  | 0.9663             | 0.9497        |

</p>

## Object Detection (Weeks 3-4)

<p align="justify"><b>Overview:</b>
We used for object detection the network YOLO and we implemented the RetinaNet based on the paper Focal Loss for Dense Object Detection.<br/>
YOLO or You Only Look Once consists on dividing the input image into an S x S grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object. Each grid cell predicts B bounding boxes and confidence scores for those boxes, reflecting how confidence and accurate the model is to that prediction.<br/>
On the other hand, the RetinaNet is a one stage detector network that matches the state-of-the-art results of two-stage detectors.The authors of this network have identified that the cause of obtaining lower accuracies comes from class imbalance in the datasets. So, they propose a dynamically scaled cross-entropy loss that down-weights the correctly classified examples and helps focusing on difficult miss-classified samples.

</p>

<p align="justify"><b>Datasets:</b>

[TT100K](http://cg.cs.tsinghua.edu.cn/traffic-sign/)<br/>
[Udacity](https://github.com/udacity/self-driving-car/tree/master/datasets)<br/>

</p>

<b>Contributions to the code:</b>
+ `code/models/retinanet.py` - RetinaNet implementation.  
+ `code/metrics/retina_metrics.py` - Focal Loss and metrics implementations for RetinaNet.  
+ `code/scripts/dataset_analysis.py` - Script for analysing the datasets.
+ `code/config/*` - Configuration files for image detection 
	
<b>Completeness of the goals:</b>

- [x] a) Run the provided code.<br> 
	- [x] Analyze the dataset.<br> 
	- [x] Compute and compare the detection F-score on the train, validation and test parts separately. <br> 
		
- [x] b) Read two papers cited in the object detection networks section.<br>  
	- [x] YOLO.<br>  
	- [x] Another paper (RetinaNet).<br> 
- [x] c) Implement a new network (c.2 - Develop the network entirely by yourself).<br> 
- [x] d) Train the networks for another dataset (Udacity).<br>
- [x] f) Report showing the achieved results.<br>



<p align="justify"><b>Results:</b>

| Neuronal Network | Dataset     |  Overall Precision | Overall Recall| Overall F1 | Average Recall| Average IoU | FPS	 |
| ---------------- |:-----------:|:------------------:|:-------------:|:----------:|:-------------:|:-----------:|:-----:|
|YOLO v2           | TT100K      | 0.6308	      | 0.3253	      | 0.4292	   | 0.9730	   | 0.7588	 | 70.11 |
|YOLO v2           | Udacity     | 0.1701	      | 0.1606	      | 0.1652	   | 0.5660	   | 0.5213	 | 68.33 |
|YOLO v2           | Udacity (Data Aug,40 epochs) | 0.1779 | 0.1911   | 0.1843	   | 0.5831	   | 0.5212	 | 67.66 |
|RetinaNet         | TT100K      | 0.5930	      | 0.0547	      | 0.1205	   | 0.9621	   | 0.7405	 | 69.22 |
|RetinaNet         | Udacity     | 0.1216	      | 0.0978	      | 0.1147	   | 0.1580	   | 0.3424	 | 66.54 |
</p>

## Semantic Segmentation (Weeks 5-6)

<p align="justify"><b>Overview:</b>
The goal of the image semantic segmentation is to label each pixel of the input image with the class that belongs. In order to do that task we used Wide - Resnet and FCN8 nets.
<br/>

</p>

<p align="justify"><b>Datasets:</b>

[CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)<br/>
[Cityscapes](https://www.cityscapes-dataset.com/)<br/>

</p>

<b>Contributions to the code:</b>
+ `code/models/wide_resnet.py` - ResnetFCN implementation.  
+ `code/config/*` - Configuration files for semantic segmentation 
	
<b>Completeness of the goals:</b>

- [x] a) Run the provided code.<br> 
- [x] b) Read two papers.<br>  
	- [x] Summary of “Fully Convolutional Networks for Semantic Segmentation”.<br>  
	- [x] Another paper (ResnetFCN).<br> 
- [x] c) Implement a new network (c.2 - Develop the network entirely by yourself).<br> 
- [x] d) Train the networks for another dataset (Cityscapes).<br>
- [x] f) Report showing the achieved results.<br>



<p align="justify"><b>Results:</b>

| Neuronal Network | Dataset     |  Accuracy | Loss | Jaccard Coefficient | FPS |
| ---------------- |:-----------:|:---------:|:----:|:-------------------:|:---:|
|FCN8              | CamVid      |0.9226     |0.2136|0.6561               |20.51| 
|FCN8              | Cityscapes  | 	     | 	    | 	                  | 	| 
|Wide ResNet       | CamVid      |0.9168     |0.3357|0.6016               |16.04|
|Wide ResNet       | Cityscapes  |	     | 	    | 	                  | 	| 
</p>
