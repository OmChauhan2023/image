# FaceID: Facial Recognition for 5 Individuals

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

**FaceID** is an end-to-end facial recognition project that classifies images of **5 individuals** using a custom-trained Convolutional Neural Network (CNN).  
The project covers everything from dataset preprocessing to training, evaluation, saving/loading the model, and making predictions on new images.

---

### ✨ Core Features

- **Multi-Class Facial Recognition**: Classifies faces into one of 5 known individuals.  
- **Data Augmentation**: Improves generalization by applying random transformations.  
- **Deep Learning Backbone**: A custom-built **CNN** architecture using TensorFlow/Keras.  
- **Train/Test Split**: Organized dataset with clear separation for robust evaluation.  
- **Prediction on New Images**: Load a trained model and classify unseen images.  

---

### 🛠️ Tech Stack & Pipeline

This project is built on modern deep learning and computer vision tools. The pipeline is structured as follows:

1. **Dataset Preparation**
   - Images are organized into folders:  
     ```
     dataset/
     ├── train/
     │   ├── person1/ person2/ ... person5/
     └── test/
         ├── person1/ person2/ ... person5/
     ```
   - Each folder contains face images of that person.

2. **Data Preprocessing (`ImageDataGenerator`)**
   - Rescales pixel values (0–1 range).
   - Applies **augmentation** (rotation, zoom, flips) to improve robustness.

3. **Model Architecture (`TensorFlow/Keras`)**
   - Custom **CNN** with:
     - Convolutional layers (feature extraction)
     - MaxPooling layers (downsampling)
     - Dense layers (classification)
     - Softmax output for **5 classes**

4. **Training & Evaluation**
   - Loss: `categorical_crossentropy`
   - Optimizer: `Adam`
   - Metrics: `Accuracy`
   - Validation performance tracked per epoch.
   - Accuracy curves plotted for training vs validation.

5. **Model Saving & Loading**
   - Save as `.h5`:
     ```python
     model.save('facial_recognition_model.h5')
     ```
   - Load for later use:
     ```python
     from tensorflow.keras.models import load_model
     model = load_model('facial_recognition_model.h5')
     ```

6. **Prediction on New Images**
   - Upload an image, preprocess to `(150x150)`, normalize, and classify.
   - Example:
     ```python
     img = image.load_img('test_face.jpg', target_size=(150,150))
     img_array = image.img_to_array(img) / 255.0
     img_array = np.expand_dims(img_array, axis=0)
     prediction = model.predict(img_array)
     print("Predicted:", labels[np.argmax(prediction)])
     ```

---

### 📊 Results
- Achieved **~XX% accuracy** on test data after 10 epochs.  
- Successfully distinguishes between 5 individuals with minimal overfitting.  

*(Add confusion matrix or accuracy screenshots here)*

---

### 🚀 Future Improvements
- Use **transfer learning** (MobileNet, FaceNet, VGGFace2) for higher accuracy.  
- Add **face detection (MTCNN / Haar cascades)** before classification.  
- Deploy as a **Flask/Streamlit web app** for real-world use.  

---

### 👨‍💻 Author
- Your Name ([@OmChauhan2023](https://github.com/OmChauhan2023)

---
