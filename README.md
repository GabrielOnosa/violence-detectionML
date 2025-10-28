# Violence Classifier Project 



Detects violent actions in videos using a combination of \*\*YOLOv8-Pose\*\*, \*\*ResNet-based feature extractor\*\* and text-embeddings from Gemini captions. Part of research on multimodal pipelines with Convolution Neural Networks during my internship at CAMPUS institute. Compared to unimodal CNN, this pipeline obtained roughly +10% (80% compared to 70%) accuracy on RWF-2000 dataset and +2% gain compared to actual Gemini captions.



---


##  Overview

The pipeline works as follows:

2\. Detect human poses using YOLOv8-Pose.

3\. Select 3 frames most likely to show violence using keypoints from YOLOv8-Pose.

4\. Run the 3 frames through the ResNet feature extractor (Each frames is grayscaled and inputted as one of the RGB channels). 

5\. Concatenate embeddings from ResNet, Yolov8-Pose and Gemini. 

6\. Run the embeddings through a classifier (either nn.Linear or Support Vector Machine)


---



\## Setup

```bash

\# Clone the repository

git clone https://github.com/yourusername/violence-detection.git

cd violence-detection



\# Install required packages

pip install -r requirements.txt

