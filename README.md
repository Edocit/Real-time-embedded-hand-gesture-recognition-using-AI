# Real-time Hand gesture recognition using AI
 This project provides a real-time implementation of three neural networks used to detect 18 different hand gestures

-----

<h2>Dataset</h2>
<img src=/imgs/hagrid_gestures.png>
We used an ad-hoc modified version of the HaGRID dataset. It includes 552.992 FullHD (1920×1080) RGB images divided into 18 classes of gestures, reported in the figure above, for a total size of 716 GB. The data is split into a training set and a test set, with 509.323 images for training (92%) and 43.669 images for testing (8%). On these images we applied data augmentation strategies to improve the number and the quality of the images

-----

<h2>Application structure</h2>
<img src=/imgs/application.png>


-----

<h2>Communication channel - Packet structure</h2>
<img src=/imgs/packet_struct.png>
<img src=/imgs/packet_burst.png>
