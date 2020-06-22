# Virtual Makeup - Apply Lipstick

This is a demonstration of how to use OpenCV and Dlib to apply lipstick to a face image, also a project for my online OpenCV course: Computer Vision II. Simple but fun!

## The Core Idea

- Detect landmarks on the face
- Fill the upper and lower lips with the lipstick color
- Generate a blurred lip mask
- Alpha blend the mask image with the lip-colored image for a more natural looking

## Code

### Import libraries and set up image display parameters

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

### Read Image Without Makeup

The original photo without makeup came from https://generated.photoswas and was generated completely by AI.

```
im = cv2.imread("AI-no-makeup.jpg")
# Convert BGR image to RGB colorspace for a correct Matplotlib display. 
# This is because OpenCV uses BGR format by default whereas Matplotlib assumes RGB format by default. 
imDlib = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.imshow(imDlib)
ax = plt.axis('off')
```

### Load landmark detector

Load Dlib’s face detector and the pre-trained 68-Point face landmark model. You can download the model file from Dlib website.

```
# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
```

### Detect faces in the image

```
faceRects = faceDetector(imDlib, 0)

```

### Detect landmarks

```
# To store landmark points
points = []
# For this simple application,
# choose the first detected face for landmark detection.
# Multiple detected faces situation not included.
if len(faceRects) > 0:
    newRect = dlib.rectangle(int(faceRects[0].left()),
                             int(faceRects[0].top()),
                             int(faceRects[0].right()),
                             int(faceRects[0].bottom()))
    # Detect landmarks
    landmarks = landmarkDetector(imDlib, newRect)

    # Convert Dlib shape detector object to list of tuples
    # and store them for 
    for p in landmarks.parts():
        pt = (p.x, p.y)
        points.append(pt)
    print(points)
else:
    print('No face detected')
```

The image below shows the detected 68 landmarks and their corresponding indices.

### Fill the upper and lower lips with lipstick color

```
# Make a copy of the original image
imLipStick = imDlib.copy()
# Get lip points from the 68 landmarks
lipPoints = points[48:68]
# Fill lip with red color. You can use any lipstick color you like.
cv2.fillPoly(imLipStick, [np.array(lipPoints)], (255, 0, 0))
plt.figure(figsize=(20,20))
plt.imshow(imLipStick)
```

You can see the image applied with lipstick by fillPoly function looks not very natural. To fix this, a blurred mask is created for alpha blending.

### Generate a Lip Mask from Lip Points
def removePolygonFromMask(mask, points, pointsIndex):
  hullPoints = []
  for pIndex in pointsIndex:
    hullPoints.append(points[pIndex])

  cv2.fillConvexPoly(mask, np.int32(hullPoints), (0, 0, 0))

# Generate a Lip Mask from Lip Points

First we need to define some functions

```
# Function for removing the lip gap from the lip mask
def removePolygonFromMask(mask, points, pointsIndex):
  hullPoints = []
  for pIndex in pointsIndex:
    hullPoints.append(points[pIndex])
  cv2.fillConvexPoly(mask, np.int32(hullPoints), (0, 0, 0))

# Function to generate the lip mask
def getLipMask(size, points):
  # Lip Gap polygon
  # Note the indices are for the lip points array.
  # Don't get them confused with the 68 landmark array indices.
  lipGap = [12, 13, 14, 15, 16, 17, 18, 19]

  # Find Convex hull of all points
  hullIndex = cv2.convexHull(np.array(points), returnPoints=False)

  # Convert hull index to list of points
  hullInt = []
  for hIndex in hullIndex:
    hullInt.append(points[hIndex[0]])

  # Create mask such that convex hull is white
  mask = np.zeros((size[0], size[1], 3), dtype=np.uint8)
  cv2.fillConvexPoly(mask, np.int32(hullInt), (255, 255, 255))

  # Remove lip gap from the mask
  removePolygonFromMask(mask, points, lipGap)

  return mask

# Run the above functions to create the lip mask
mask = getLipMask(imLipStick.shape[0:2], lipPoints)
```

### Blur the lip mask to smooth edges for a more natural lipstick effect

```
maskHeight, maskWidth = mask.shape[0:2]
maskSmall = cv2.resize(mask, (256, int(maskHeight*256.0/maskWidth)))
maskSmall = cv2.erode(maskSmall, (-1, -1), 25)
maskSmall = cv2.GaussianBlur(maskSmall, (51, 51), 0, 0)
mask = cv2.resize(maskSmall, (maskWidth, maskHeight))
```

### Alpha blend to apply lipstick

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

Compare the two images before and after alpha blending with the blurred mask, the latter looks more natural. Of course, there are many other ways to achive more sophisticated results, but for a simple application the current method is a shortcut.
