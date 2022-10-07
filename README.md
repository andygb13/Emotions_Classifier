# Emotions_Classifier
Emotions classifier using machine learning based on eye images.

<img src="https://github.com/andygb13/Emotions_Classifier/blob/main/Samples/Picture1.png"  width="500" height="500">

This repository includes a pipeline for processing facial images and training an emotions classifier. The dataset used for this project was the AffectNet dataset, which includes more than 30k training samples, and 3k evaluation samples. The images to be processed are cropped (periocular area) facial images from the original AffectNet dataset, these were generated using a YOLOv5 object detector. The processed images are used for training a ResNet-50 model, this model's inteded use is to predict an emotion from a periocular area image, as shown in the example below:

<img src="https://github.com/andygb13/Emotions_Classifier/blob/main/Samples/Picture2.png"  width="500" height="500">


## File Structure


## How to use live implementation

### Requirements
