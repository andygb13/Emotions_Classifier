# Periocular Emotions Classifier
Emotions classifier using machine learning based on eye images (the periocular area of the face).

<img src="https://github.com/andygb13/Emotions_Classifier/blob/main/Samples/Picture1.png"  width="500" height="500">

This repository includes a pipeline for processing facial images and training an emotions classifier. The dataset used for this project was the AffectNet dataset, which includes more than 30k training samples, and 3k evaluation samples. The images to be processed are cropped and augmented (eyes) facial images from the original AffectNet dataset, these were generated using a YOLOv5 object detector:

<img src="https://github.com/andygb13/Emotions_Classifier/blob/main/Samples/cropped_img.png"  width="800" height="150">

The processed images are used for training a ResNet-50 model. This model's inteded use is to predict an emotion from a periocular area image, as shown in the example below:

<img src="https://github.com/andygb13/Emotions_Classifier/blob/main/Samples/Picture2.png"  width="500" height="500">

A live implementation script live_emotions.py is included for showcasing the object detection and emotions classifier live using a local webcam (or desired source). See below for details on how to use live implementation.

## File Structure

    ├── Processing and Training      # Jupyter Notebook files detailing the image and training pipelines
    |    ├── Dataset_Processing.ipynb     # Data processing to use for training classifier
    |    ├── Emotions_Classifier.ipynb    # Training of a ResNet-50 model for emotion classification 
    ├── Samples                      # Samples from final product on stock images
    ├── live_emotions.py             # Python program to perform a live implementation of both the object detector and classifier
    └── README.md

## How to use live implementation

Prior to using the live_emotions script make sure the libraries imported in the first eight lines of the scrip have been installed. Also, install yolov5 requirements which are included as a comment in the script. Once these have been installed, replace the classifier and object detector path (cl_path and od_path) with the location of the files in your local computer. The script should now be ready to run.

<img src="https://github.com/andygb13/Emotions_Classifier/blob/main/Samples/mask.png"  width="400" height="300">

The live_emotions.py script has been tested using versions of Python 9 and Python 10. 

### Requirements

The files for the Tensorflow emotions classification model and the Pytorch object detector weights are not included in this repository due to file size constraints. Please email andygb13@gmail.com to to request the files and they will be sent to you as soon as possible.

The live implementation will not work if a webcam (or other image source) is not connected to the local computer.
