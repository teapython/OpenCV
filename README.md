# DoppelGanger - Find Celebrity Look-Alike

This is a demonstration of using deep learning based face recognition to find a doppelganger or look-alike celebrity to a given person, also one of my projects for the OpenCV course: **Computer Vision II**. 

## The Core Idea

- Enrollment of celebrity images in Dlib’s pre-trained Face Recognizer neural network
    * Process each celebrity image to detect faces and face landmarks
    * Compute face descriptor for each image
- Process a given person' image in the same way to compute face descriptor 
- Calculate Euclidean distance between face descriptor of the person versus face descriptors of celebrity images. Find the celebrity face for which distance is minimum as the look-alike face

## Code

Import libraries and set up image display parameters.

```
import cv2, dlib
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

Initialize Dlib’s Face Detector, Facial Landmark Detector and Face Recognition neural network objects. 
All the models can be found through this blog post of Davis King, the author of Dlib.

```
faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
faceRecognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
```

A mini celebrity dataset from OpenCV is used for this project. It's ~400MB in size, including 5 images per person for ~1100 celebs. You can use any other large datasets if you have enough computation resources. 

The **celeb_mini** dataset folder has the following structure:

```
celeb_mini
└───n00000001
│   └──n00000001_00000263.JPEG
│   └──n00000001_00000405.JPEG
│      ...
└───n00000003
│   └──n00000003_00000386.JPEG
│   └──n00000003_00000488.JPEG
│       ...
│
```

A dictionary **labelMap** contains the mapping between the subfolder names and the celebrity's actual name as show below:

```
{'n00000001': 'A.J. Buckley',
 'n00000002': 'A.R. Rahman',
 'n00000003': 'Aamir Khan',
 'n00000004': 'Aaron Staton',
 'n00000005': 'Aaron Tveit',
 'n00000006': 'Aaron Yoo',
 'n00000007': 'Abbie Cornish',
 .
 .
 .
}
```

Now prepare the dataset images for enrollment in the Dlib's neural network.

```
# imagePaths is a list of full paths for all celeb images  
# nameLabelMap is a dictionary with keys as image file full paths
# and values as subfolder names containing these images for a celebrity

nameLabelMap = {}
imagePaths = []

subfolders = os.listdir(faceDatasetFolder)

for subfolder in subfolders:
    xpath = os.path.join(faceDatasetFolder, subfolder)
    if os.path.isdir(xpath):
        for x in os.listdir(xpath):
            fullPath = os.path.join(xpath, x)
            # Change 'JPEG' to the real format of your images
            if x.endswith('JPEG'):
                imagePaths.append(fullPath)
                nameLabelMap[fullPath] = subfolder.split('/')[-1]
```

The dictionary **nameLabelMap** looks like this:

```
{'celeb_mini/n00001021/n00001021_00000223.JPEG': 'n00001021', 
 'celeb_mini/n00001021/n00001021_00000242.JPEG': 'n00001021',
 .
 .
 .
}
```

Process images one by one through face detection and face landmark detection. Then compute face descriptors through the Dlib's neural network. Note this step may take a while depending on your system.

```
# Store face descriptors in an ndarray (faceDescriptors)
# and their corresponding subfolder names in a dictionary (index)
index = {}
i = 0
faceDescriptors = None
for imagePath in imagePaths:
    # Read image and convert it to RGB as Dlib uses BGR as default format
    im = cv2.imread(imagePath)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Detect faces in image
    faces = faceDetector(img)
    
    # Process each face found
    for k, face in enumerate(faces):
        # Find facial landmarks for each detected face
        shape = shapePredictor(img, face)
        
        # Compute face descriptor using neural network defined in Dlib.
        # It is a 128D vector that describes the face in img identified by shape.
        faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)
        
        # Convert face descriptor from Dlib's format to list, then a NumPy array
        faceDescriptorList = [x for x in faceDescriptor]
        faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
        faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]
        
        # Stack face descriptors (1x128) for each face in images, as rows
        if faceDescriptors is None:
            faceDescriptors = faceDescriptorNdarray
        else:
            faceDescriptors = np.concatenate((faceDescriptors, faceDescriptorNdarray), axis=0)
        
        # Save subfolder names containing this face image in index
        # Later it will be used to identify person name corresponding to face descriptors
        index[i] = nameLabelMap[imagePath]
        i += 1
```

The dictionary **index** looks like this:

```
{0: 'n00001021', 
 1: 'n00001021', 
 2: 'n00001021', 
 .
 .
 .
}
```

Now we can use minimum distance rule to find the closest celeb in the celeb dataset to a given person's image. We use images of **Sofia Solares** and **Shashikant Pedwal** as examples.

The example images processing steps, including face detection, face landmark detection, and face descriptor computation, are the same as the previous enrollment so not repeated here. But note there's only one output **faceDescriptorNdarray** instead of a stacked array for each example image, because we only consider the situation of single face per image.

```
# Same code as previous enrollment steps except for reading example images
... 
# faceDescriptorNdarray here is for the example image
faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :] 
        
# Calculate Euclidean distances between face descriptor calculated on face dectected
# in the example image with all the face descriptors we calculated while enrolling faces
faceDescriptorsEnrolled = faceDescriptors
distances = np.linalg.norm(faceDescriptorsEnrolled - faceDescriptorNdarray, axis=1)
        
# Calculate minimum distance and index of this face
argmin = np.argmin(distances)  # index
minDistance = distances[argmin]  # minimum distance

# The face with the minimum distance is the celeb look-alike face
# Find the full path of the image in nameLabelMap
imagePath = list(nameLabelMap.keys())[list(nameLabelMap.values()).index(index[argmin])]  
# Find the name of person from dictionary labelMap based on index 
celeb_name = labelMap[index[argmin]]
```

## Final Results

![Shashikant Pedwal](/data/images/shashikant-pedwal.jpg#center)
![](/data/images/sofia-solares.jpg) ![](/data/images/alike1.JPEG)  ![](/data/images/alike2.JPEG)    
