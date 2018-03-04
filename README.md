# Scene Understanding for Autonomous Vehicles

### Master in Computer Vision - M5 Visual recognition

## Team "Avengers"
- Mikel Menta Grade - mikel.menta@e-campus.uab.cat
- Alex Vallès Fernández - alex.valles@e-campus.uab.cat
- Sebastian Maya Hernández - sebastiancamilo.maya@e-campus.uab.cat
- Pedro Luis Trigueros Mondéjar - pedroluis.trigueros@e-campus.uab.cat

<p align="justify"><b>Overview:</b> 
In this project we focus on scene understanding for autonomous vehicles. Understanding the context of the own vehicle is key for autonomous driving. The project consists of three parts or stages, corresponding to object detection, recognition/classification and semantic segmentation.<br/>
Furthermore, we aim to learn the basic concepts, techniques, tricks and libraries to develop and evaluate deep neural networks.
</p>

### WEEK2
<p align="justify"><b>Code explanation:</b>
We have implemented the winner architecture of ILSVRC 2017 Classification Competition, the Squeeze-Excitation ResNet. 
The squeeze-excitation block consists on an squeeze step based on a Global Average Pooling over the output of the residual block and afterwards, an excitation step based on obtaining some weights for each output channel of the residual block and multiplying the channels by those weights. To obtain those weights, two fully-connected layers are used. The first one performs a dimensionality reduction over the number of channels C and uses ReLU activation. The reduction has been performed with a reduction ratio of r=16. The second FC layer  recovers the original dimensionality C and uses sigmoid activation for obtaining a weight in range [0,1].
</p>

<p align="justify"><b>Results:</b>


| Neuronal Network         | Dataset     | Accuracy training  | Accuracy test |
| ------------------------ |:-----------:| ------------------:|:-------------:|
| VGG                      | Ttk100      | 0.9664             | 0.8546        |
| VGG                      | BelgiumTSC  | 0.9875             | 0.9607        |
| queeze-Excitation ResNet | Ttk100      | 0.9987             | 0.9619        |
|queeze-Excitation ResNet  | BelgiumTSC  | 0.9978             | 0.9655        |

</p>

<p align="justify"><b>Completeness of the goals:</b>
100% completed
</p>

## Weights of the model
*   **Week 2:** Download [here](https://drive.google.com/file/d/1Jpp32Rv_DRf0ml6YI4snDIONUjovq30b/view?usp=sharing)
*   **Week 3:**
*   **Week 4:**
*   **Week 5:**

## Instructions for use code 
*   **Week 2:** python train.py -c config/configFile.py -e expName
*   **Week 3:**
*   **Week 4:**
*   **Week 5:**

## Project Schedule
*   **Week 1:** [Project introduction](https://www.google.es)
*   **Week 2:** [Slides](https://docs.google.com/presentation/d/1rJLKQuK2cMjYeAlZBgj0RB2ovhd1AvWGDzMVq4Z844E/edit?usp=sharing)
*   **Week 3:**
*   **Week 4:**
*   **Week 5:**

## Documentation
- [Overleaf article](https://www.overleaf.com/read/fbhxbjqydfwx)

### Papers
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](https://www.overleaf.com/read/jwgpqfzvnwgk)
- [Squeeze and Excitation Blocks](https://www.overleaf.com/read/bxmpwjqhckmn)

## Peer review
- Intra-group evaluation form for Week 2 (T.B.A.)
- Intra-group evaluation form for Weeks 3/4 (T.B.A.)
- Intra-group evaluation form for Weeks 5/6 (T.B.A.)
- Intra-group evaluation form for Week 7 (T.B.A.)
