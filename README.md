# Real-time Hand gesture recognition using AI
 This project provides a real-time implementation of three neural networks used to detect 18 different hand gestures

<img src=/imgs/demo.gif>


<h2>Dataset</h2>
<img src=/imgs/hagrid_gestures.png>
We used an ad-hoc modified version of the HaGRID dataset. It includes 552.992 FullHD (1920×1080) RGB images divided into 18 classes of gestures, reported in the figure above, for a total size of 716 GB. The data is split into a training set and a test set, with 509.323 images for training (92%) and 43.669 images for testing (8%). On these images we applied data augmentation strategies to improve the number and the quality of the images

<h2>Application structure</h2>
<img src=/imgs/application.png>
The processing pipeline is composed of three stages:

1) **Image acquisition and preprocessing**: The purpose of this stage is to fetch the frame from the camera and adapt the image for the neural network. The blocks involved in this stage are visualized in orange on the left of the figure above.

2) **CNN inference**: This stage (corresponding to the yellow CNN block) performs the inference of the neural network on the rescaled input image, producing an output vector of 18 elements (y_0, …, y_17), corresponding to the number of different categories of gestures present in the training set.

3) **Post-processing subsystem**: The purpose of this stage is to select the gesture recognized by the network based on all the confidence scores and reduce possible fluctuations in the outputs produced by the network on a sequence of input images. The blocks involved in this stage are the ones included in the purple box. 

_Notice that in GPU-accelerated systems, like the one used in this project, the only part that benefits from hardware acceleration is the CNN inference. Using heterogeneous platforms with FPGAs it is also possible to accelerate the preprocessing step and the filtering system, by implementing dedicated hardware devices in programmable logic. However, FPGA acceleration introduces several complications in the deployment of a neural network that are out of the scope of this document._



-----

<h2>Communication channel - Packet structure</h2>
<img src=/imgs/packet_struct.png>
<img src=/imgs/packet_burst.png>
