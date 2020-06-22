# Virtual Makeup - Apply Lipstick

This is a demonstration of how to use OpenCV and Dlib to apply lipstick to a face image, also a project for my attended online OpenCV course: Computer Vision II. Simple but fun!

## The Core Idea

- Detect landmarks on the face
- Fill the upper and lower lips with the lipstick color
- Generate a blurred lip mask
- Alpha blend the mask with the lip-colored image for a more natural looking

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

There are 68 landmarks on a face, but for applying lipstick we only care about the lip points (48 to 67). Now fill the upper and lower lips with lipstick color you like. 

```
# Make a copy of the original image
imLipStick = imDlib.copy()
# Get lip points from the 68 landmarks
lipPoints = points[48:68]
# Fill lip with red color
cv2.fillPoly(imLipStick, [np.array(lipPoints)], (255, 0, 0))
```

You can see the applied lipstick using fillPoly function has rough and sharp edges. To get a more natural looking, a heavily blurred lip mask is used.

![](/data/images/face_with_simple_lipstick.jpg)

First we need to define some functions for generating the lip mask.

```
# Function to remove the lip gap from the lip mask
def removePolygonFromMask(mask, points, pointsIndex):
  hullPoints = []
  for pIndex in pointsIndex:
    hullPoints.append(points[pIndex])
  cv2.fillConvexPoly(mask, np.int32(hullPoints), (0, 0, 0))

# Function to generate the lip mask
def getLipMask(size, points):
  # Find Convex hull of all points
  hullIndex = cv2.convexHull(np.array(points), returnPoints=False)

  # Convert hull index to list of points
  hullInt = []
  for hIndex in hullIndex:
    hullInt.append(points[hIndex[0]])

  # Create mask such that convex hull is white
  # Note this mask also includes the lip gap 
  # which will be removed
  mask = np.zeros((size[0], size[1], 3), dtype=np.uint8)
  cv2.fillConvexPoly(mask, np.int32(hullInt), (255, 255, 255))

  # Remove lip gap from the mask
  removePolygonFromMask(mask, points, lipGap)

  return mask
```

Use these functions to create the lip mask:

```
mask = getLipMask(imLipStick.shape[0:2], lipPoints)
```

![](/data/images/mask_before_blur.jpg)

Now blur the lip mask to smooth edges for a more natural lipstick effect.

```
maskHeight, maskWidth = mask.shape[0:2]
maskSmall = cv2.resize(mask, (256, int(maskHeight*256.0/maskWidth)))
maskSmall = cv2.erode(maskSmall, (-1, -1), 25)
maskSmall = cv2.GaussianBlur(maskSmall, (51, 51), 0, 0)
mask = cv2.resize(maskSmall, (maskWidth, maskHeight))
```

The lip mask looks heavily blurred.

![](/data/images/mask_after_blur.jpg)

Next step is to use the blurred mask to alpha blend the original image with the lipstick applied image.

```
# Define the function for alpha blending
def alphaBlend(alpha, foreground, background):
  fore = np.zeros(foreground.shape, dtype=foreground.dtype)
  fore = cv2.multiply(alpha, foreground, fore, 1/255.0)

  alphaPrime = np.ones(alpha.shape, dtype=alpha.dtype)*255 - alpha
  back = np.zeros(background.shape, dtype=background.dtype)
  back = cv2.multiply(alphaPrime, background, back, 1/255.0)

  outImage = cv2.add(fore, back)
  return outImage
 
# Perform alpha blending
imLipStick = alphaBlend(mask, imLipStick, imDlib)
```

## Final Results
Compare the two images before and after alpha blending with the blurred mask, the latter looks more natural. Of course, there are many ways to achive more sophisticated results, but for a simple application like this the current method is a shortcut.

![](/data/images/face_with_simple_lipstick.jpg) ![](/data/images/face_with_natural_lipstick.jpg)
