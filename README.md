# Posit-based DNN for autonomous driving
Repository of the project for the course of Symbolic and Evolutionary Artificial Intelligence, AIDE unipi.
In this project, we compared the performance (accuracy and inference time) of neural networks based on posit 
with those in other numerical formats. The tests were conducted with a classification task on the GTSDB 
dataset and an object detection task using the GTSRB dataset.

## Classification 
The classification was performed using both from-scratch and pretrained neural networks, specifically 
ResNet50v2 and MobileNet. We compared the posit format with the float16, float32, and float64 formats.

## Object detection 
In the object detection task, we employed RetinaNet, modifying the original code to make it compatible 
with the posit format. Subsequently, we compared the Intersection Over Union (IoU) of the networks in 
different numerical formats to assess their performance.
