# Scene Understanding for Autonomous Vehicles

### Master in Computer Vision - M5 Visual recognition

## Team "Avengers"
- Mikel Menta Grade - mikel.menta@e-campus.uab.cat
- Alex Vallès Fernández - alex.valles@e-campus.uab.cat
- Sebastian Maya Hernández - sebastiancamilo.maya@e-campus.uab.cat
- Pedro Luis Trigueros Mondéjar - pedroluis.trigueros@e-campus.uab.cat

<p align="justify"><b>Project overview:</b>
	
In this project we focus on scene understanding for autonomous vehicles. Understanding the context of the own vehicle is key for autonomous driving. The project consists of three parts or stages, corresponding to object detection, recognition/classification and semantic segmentation.<br/>
Furthermore, we aim to learn the basic concepts, techniques, tricks and libraries to develop and evaluate deep neural networks.
</p>

<p align="justify"><b>Datasets:</b>

[TT100K](http://cg.cs.tsinghua.edu.cn/traffic-sign/)<br/>
[BelgiumTSC](http://btsd.ethz.ch/shareddata/)<br/>
[KITTI](http://www.cvlibs.net/datasets/kitti/)<br/>

</p>

### Documentation
- [Overleaf article](https://www.overleaf.com/read/fbhxbjqydfwx)
- [Google Slides](https://docs.google.com/presentation/d/1rJLKQuK2cMjYeAlZBgj0RB2ovhd1AvWGDzMVq4Z844E/edit?usp=sharing)

### Weights of the model
The weights, experiments' info and the TensorBoard logs are available [here](https://drive.google.com/drive/folders/15mccIEOhimkFo_5K3M43Vmu9xT0CEvpC?usp=sharing).

### Papers
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://www.overleaf.com/read/jwgpqfzvnwgk)
- [Squeeze and Excitation Blocks](https://www.overleaf.com/read/bxmpwjqhckmn)
- [You Only Look Once (YOLO)](https://www.overleaf.com/read/fvtfvvwbptsv)

## Object Recognition (Week 2)

<p align="justify"><b>Code explanation:</b>
We have implemented the winner architecture of ILSVRC 2017 Classification Competition, the Squeeze-Excitation ResNet. 
The squeeze-excitation block consists on an squeeze step based on a Global Average Pooling over the output of the residual block and afterwards, an excitation step based on obtaining some weights for each output channel of the residual block and multiplying the channels by those weights. To obtain those weights, two fully-connected layers are used. The first one performs a dimensionality reduction over the number of channels C and uses ReLU activation. The reduction has been performed with a reduction ratio of r=16. The second FC layer  recovers the original dimensionality C and uses sigmoid activation for obtaining a weight in range [0,1].
</p>

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

<b>Completeness of the goals:</b>

- [ ] a) Run the provided code.<br> 
	- [ ] Analyze the dataset.<br> 
	- [ ] Compute and compare the detection F-score on the train, validation and test parts separately. <br> 
		
- [ ] b) Read two papers cited in the object detection networks section.<br>  
	- [ ] YOLO.<br>  
	- [ ] Another paper.<br> 
- [ ] c) Implement a new network.<br> 
- [ ] d) Train the networks for another dataset (Udacity).<br>
- [ ] e) Try to boost the performance of your network.<br>
- [ ] f) Report.<br>


## Instructions for use code 
*   **Week 2:** python train.py -c config/configFile.py -e expName
*   **Week 3:**
*   **Week 4:**
*   **Week 5:**
