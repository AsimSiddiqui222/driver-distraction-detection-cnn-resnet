# driver-distraction-detection-cnn-resnet
A study on driver distraction detection using deep learning, comparing CNN and ResNet50 models with transfer learning.
---
## Models Used
- **Convolutional Neural Network (CNN)** – baseline model  
- **ResNet50 (Frozen)** – transfer learning without fine-tuning  
- **ResNet50 (Fine-Tuned)** – transfer learning with fine-tuning
---
## Dataset
- **Name** Distracted Driver Detection Computer Vision Dataset
- **Source:** Roboflow
- **Link**: https://universe.roboflow.com/sample-fqpfe/distracted-driver-kk1pl
- **Description:** The dataset contains images of drivers performing various activities such as safe driving, texting, talking on the phone, drinking, and interacting with passengers.
### Dataset Details
- **Total Classes:** 10  
- **Training Samples:** 1875 images  
- **Validation Samples:** 201 images  
- **Test Samples:** 112 images

### Classes:
- drinking  
- hair and makeup  
- operating the radio  
- reaching behind  
- safe driving  
- talking on the phone (left)  
- talking on the phone (right)  
- talking to passenger  
- texting (left)  
- texting (right)
- The dataset shows slight class imbalance across categories.

---
## Preprocessing
- Images resized to **224 × 224 pixels**  
- Pixel normalization applied  
- Dataset organized into training, validation, and test sets  

---

## Results
- The **CNN model achieved the highest accuracy (85.7%)**  
- The **ResNet50 (Frozen)** model achieved **73% accuracy**  
- The **ResNet50 (Fine-Tuned)** model improved performance to **82% accuracy**  
- Fine-tuning improved performance, but the baseline CNN performed best on this dataset  
- Misclassification occurred between visually similar classes such as texting and talking on the phone
  

---
## Live Demo 
- A real-time driver distraction detection demo is included using OpenCV.
- Run The Demo using live_demo.py, change cap = cv2.VideoCapture(0) to cap = cv2.VideoCapture(1) to run an external Camera
- the demo has 3 models, press 1 for CNN, press 2 for ResNet50 (Frozen), press 3 for ResNet50 (Fine-Tuned) 
- live_demo.py
---

##  How to Run

### 1. Install dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn pandas opencv-python

### Requirments
-tensorflow
-numpy
-pandas
-matplotlib
-seaborn
-scikit-learn
-OpenCV-Python
