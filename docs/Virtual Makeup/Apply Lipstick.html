## Virtual Makeup - Apply Lipstick
This is a demonstration of how to use OpenCV and Dlib to apply lipstick to a face image, also a project for my online OpenCV course: Computer Vision II. Simple but fun!

### The Core Idea

- Detect landmarks on the face
- Fill the upper and lower lips with the lipstick color
- Generate a blurred lip mask 
- Alpha blend the mask image with the lip-colored image for a more natural looking

### Code and Results

#### Import libraries and set up image display parameters
```
import cv2,sys,dlib,time,math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0,8.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'
```

#### Read Image Without Makeup
The original photo without makeup came from https://generated.photoswas and was generated completely by AI.
```
im = cv2.imread("AI-no-makeup.jpg")
# Convert BGR image to RGB colorspace for a correct Matplotlib display. 
# This is because OpenCV uses BGR format by default whereas Matplotlib assumes RGB format by default. 
imDlib = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.imshow(imDlib)
ax = plt.axis('off')
```

#### Load landmark detector
Load Dlib's face detector and the 68-Point face landmark model. You can download the model file from Dlib website. Note: Here I used Dlib's pre-trained models for face and face landmark detection, but you can train your own models.
```
# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
```
