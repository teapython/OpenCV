# Virtual Makeup - Apply Earrings

This is a demonstration of how to use OpenCV and Dlib to apply earrings to a face image, also a project for the OpenCV course: Computer Vision II. For those who love earrings!

## The Core Idea

- Detect landmarks on the face
- Scale the earring image to fit to the face
- Estimate the earlobe locations based on landmarks
- Using earlobe locations to find regions on the face image to be replaced with the earring images
- Alpha blend images of these regions with the earring images
- Replace regions in the face image with the alpha-blended images

## Code

Import libraries and set up image display parameters.

```
import cv2, dlib
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

Read the original image without makeup. It was generated completely by AI from https://generated.photos.

```
im = cv2.imread("AI_no_makeup.jpg")
# Convert BGR image to RGB colorspace for a correct Matplotlib display. 
imDlib = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.imshow(imDlib)
```
![](/data/images/AI_no_makeup.jpg)

Load Dlib’s face detector and the pre-trained 68-Point face landmark model. You can download the model file from Dlib website.

```
# Initiate the face detector instance
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
```

Detect faces in the image. The output of the Dlib face detector is a rectangle (x, y, w, h) that contains the face. 

```
faceRects = faceDetector(imDlib, 0)
```

Next step is to detect the 68 landmarks inside the detected face rectangle. 

```
# To store landmark points
points = []
# For this simple application,
# I choose the first detected face for landmark detection. 
# Multiple detected faces situation isn't considered.
if len(faceRects) > 0:
    newRect = dlib.rectangle(int(faceRects[0].left()),
                             int(faceRects[0].top()),
                             int(faceRects[0].right()),
                             int(faceRects[0].bottom()))
    # Detect landmarks
    landmarks = landmarkDetector(imDlib, newRect)

    # Convert Dlib shape detector object to list of tuples and store them.
    for p in landmarks.parts():
        pt = (p.x, p.y)
        points.append(pt)
else:
    print('No face detected')
```

The image below shows the detected 68 landmarks and their corresponding indices.

![](/data/images/face_with_landmarks.jpg)

I use the following equation to find the approximate locations of the left and right earlobe centres. This is just a rough calculation for this simple application, you can develop more accurate algorithms.

- Left: (x coordinate of landmark point 0, y coordinate of the middle point between point 2 and 3)
- Right: (x coordinate of landmark point 16, y coordinate of the middle point between point 13 and 14)

```
leftLobeCentre = (points[0][0], int(points[2][1]+(points[3][1]-points[2][1])/2))
rightLobeCentre = (points[16][0], int(points[13][1]+(points[14][1]-points[13][1])/2))
```

Now read the left and right earring images. Note they are PNG images which contain R, G, B channels, as well as a 4th channel which is a transparent alpha mask. 

```
# Read the left earring image with alpha channel
lEaringIm = cv2.imread("left_earring.png", -1)
# Split png left earring image
bl,gl,rl,al = cv2.split(lEaringIm)
# Merge and convert RGB channels into an RGB image
lEaringIm = cv2.merge((bl,gl,rl))
lEaringIm = cv2.cvtColor(lEaringIm, cv2.COLOR_BGR2RGB)
# Save the alpha information into a single mask image
lAlpha = cv2.merge((al,al,al))

# Repeat the previous steps for the right earring PNG image
rEaringIm = cv2.imread("right_earring.png", -1)
br,gr,rr,ar = cv2.split(rEaringIm)
rEaringIm = cv2.merge((br,gr,rr))
rEaringIm = cv2.cvtColor(rEaringIm, cv2.COLOR_BGR2RGB)
rAlpha = cv2.merge((ar,ar,ar))
```

The earring images and their alpha masks:
![](/data/images/left_earring.png)  ![](/data/images/left_earring_alpha.jpg) ![](/data/images/right_earring.png)  ![](/data/images/right_earring_alpha.jpg)    

Resize the earring image height to 0.8 * height difference between landmark point 2 and 4. Then resize the earring width using the same scale. Note: This scale is based on best fitting results for the current earring images. You may use a different scale for your own earring images.

```
h = int((points[4][1] - points[2][1])*0.8)
w = int(lEaringIm.shape[1]*h/lEaringIm.shape[0])
lEaringImResized = cv2.resize(lEaringIm, (w, h))
rEaringImResized = cv2.resize(rEaringIm, (w, h))

lAlphaResized = cv2.resize(lAlpha, (w, h))
rAlphaResized = cv2.resize(rAlpha, (w, h))
```

Find two rectangle regions of the exact size as the scaled earring images on the face image. The previously calculated earlobe centres are used as the top middle points of the regions.

```
# Make a copy of the original image
imEarrings = imDlib.copy()

lReg = imEarrings[points[2][1]:points[2][1]+h, int(points[0][0]-w/2):int(points[0][0]-w/2)+w]
rReg = imEarrings[points[14][1]:points[14][1]+h, int(points[16][0]-w/2):int(points[16][0]-w/2)+w]
```

Perform alpha blending for the left and right earring regions. For more detailed information about Alpha Blending, go to https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/.

```
# Convert uint8 to float and then normalize the alpha mask to keep intensity between 0 and 1
lEaringImResized = lEaringImResized.astype(float)
lReg = lReg.astype(float)
lAlphaResized = lAlphaResized.astype(float)/255

rEaringImResized = rEaringImResized.astype(float)
rReg = rReg.astype(float)
rAlphaResized = rAlphaResized.astype(float)/255

# Perform alpha blending
lForeground = cv2.multiply(lAlphaResized, lEaringImResized)
lBackground = cv2.multiply(1.0 - lAlphaResized, lReg)
lOutImage = cv2.add(lForeground, lBackground)

rForeground = cv2.multiply(rAlphaResized, rEaringImResized)
rBackground = cv2.multiply(1.0 - rAlphaResized, rReg)
rOutImage = cv2.add(rForeground, rBackground)
```

Final step is to replace the earring regions in the original image with the alpha blended images.

```
imEarrings[points[2][1]:points[2][1]+h, int(points[0][0]-w/2):int(points[0][0]-w/2)+w] = lOutImage
imEarrings[points[14][1]:points[14][1]+h, int(points[16][0]-w/2):int(points[16][0]-w/2)+w] = rOutImage
```

## Final Results

![](/data/images/face_with_earrings.jpg)
