# Real-time embedded hand gesture recognition using AI
<img align="right" width="300" src="imgs/logo_leonardo.png" />

 This project provides a real-time implementation of three neural networks used to detect 18 different hand gestures. The 3 provided models are **ResNet-50**, **MobileNet-v3**, and **Inception-v3**.



<p align="center">
 <img align="center" width="700" src=/imgs/demo.gif>
</p>


One additional "No gesture" class is added in the application to consistently handle uncertain positions or transition frames between different gestures 

<h2>Dataset</h2>
<p align="center">
 <img align="center" src=/imgs/hagrid_gestures.png>
</p>
We used an ad-hoc modified version of the HaGRID dataset. It includes 552.992 FullHD (1920×1080) RGB images divided into 18 classes of gestures, reported in the figure above, for a total size of 716 GB. The data is split into a training set and a test set, with 509.323 images for training (92%) and 43.669 images for testing (8%). On these images we applied data augmentation strategies to improve the number and the quality of the images

<h2>Application structure</h2>
<p align="center">
<img width="800" src=/imgs/application.png>
</p>
The processing pipeline is composed of three stages:

1) **Image acquisition and preprocessing**: The purpose of this stage is to fetch the frame from the camera and adapt the image for the neural network. The blocks involved in this stage are visualized in orange on the left of the figure above.

2) **CNN inference**: This stage (corresponding to the yellow CNN block) performs the inference of the neural network on the rescaled input image, producing an output vector of 18 elements (y_0, …, y_17), corresponding to the number of different categories of gestures present in the training set.

3) **Post-processing subsystem**: The purpose of this stage is to select the gesture recognized by the network based on all the confidence scores and reduce possible fluctuations in the outputs produced by the network on a sequence of input images. The blocks involved in this stage are the ones included in the purple box. 

_Notice that in GPU-accelerated systems, like the one used in this project, the only part that benefits from hardware acceleration is the CNN inference. Using heterogeneous platforms with FPGAs it is also possible to accelerate the preprocessing step and the filtering system, by implementing dedicated hardware devices in programmable logic._



<h2>Communication channel - Packet structure</h2>

The application includes the possibility to enable different PHY communication channels, including **serial ports** supporting the **RS-232** standard. The transmission mode of the serial port is **fully configurable**, however, the default setting is 8N1 (8 bit of data, no parity, and one stop bit). The data frame generated and transmitted by the application, shown bleow, consists in two bytes:
1) **Detected class**: The class detected by the neural network. It is expressed as a byte.
2) **Timing frame**: The amount of time, expressed in milliseconds _(ms)_ passed since the previous gesture sent.

<p align="center">
<img src=/imgs/packet_struct.png>
</p>

In particular, the first byte of the packet is always the class prediction, while the second is the timestamp measured at the end of the **Post-processing subsystem**, so before the serial "composite frame" is sent. (see previous section).

The use of tehe above serial frame allows the communication with external devices, which using the timestamp, can detect __gestures combo__. An example of a serial framing is reported in the figure below 

<p align="center">
<img src=/imgs/packet_burst.png>
</p>

The application was structured to meet **real-time constraints**, which in the case of camera-based applications most of the time corresponds to a processing time of 34 milliseconds (30 FPS) for each frame. However, if any error occurs, and the timestamp exceeds 255 ms, the 255 value is held until its transmission.  
