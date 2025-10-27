# 🖐️ Arabic Sign Language Recognition

An AI-powered deep learning project that recognizes **Arabic sign language letters** using computer vision and real-time inference.  
Developed to support inclusive communication and demonstrate the power of **CV + DL integration**.

---

## 🧠 Overview
This project combines **deep learning training notebooks** and **real-time camera demos** to detect and classify Arabic hand signs.  
It uses convolutional neural networks (CNNs) trained on an RGB dataset of Arabic alphabets, enhanced with **data augmentation and synthesis**.

---

## 📂 Project Structure

### 📓 Notebooks
| Notebook | Description |
|-----------|--------------|
| **`train_arabic_sign_language.ipynb`** | Main notebook for training the Arabic sign language classification model with clean architecture and preprocessing. |
| **`train_augmented.ipynb`** | Trains the same model with advanced data augmentation to improve robustness and generalization. |
| **`demo_camera_sign_recognition.ipynb`** | Real-time camera demo that recognizes hand gestures and predicts Arabic letters using the trained model. |
| **`demo_camera_dynamic_roi_hand_detection.ipynb`** | Enhanced camera demo with dynamic Region of Interest (ROI) and hand detection for more accurate gesture capture. |

---

## 🧩 Models
| Model | Description |
|--------|-------------|
| **`model_best.ckpt`** | The best-performing model trained on the original dataset. |
| **`model_best_augmented.ckpt`** | Model trained with additional augmented and synthetic data for improved accuracy and robustness. |

---

## 🖼️ Demo Images
Images captured during live demos showing recognized Arabic letters with their prediction accuracies.

<p align="center">
  <img src="demo_images/demo1.png" width="320" alt="Arabic sign demo 1">
  <img src="demo_images/demo2.png" width="320" alt="Arabic sign demo 2"><br>
  <em>Real-time detection of Arabic letters using camera input.</em>
</p>

---

## 🧰 Dataset
We used the **[RGB Arabic Alphabets Sign Language Dataset](https://www.kaggle.com/datasets/muhammadalbrham/rgb-arabic-alphabets-sign-language-dataset)** from Kaggle.

This dataset contains RGB images of Arabic sign language gestures.  
We expanded it using **Data Augmentation and Synthesis** to improve model performance:

- **Generative Adversarial Networks (GANs)** to create synthetic training samples  
- **3D simulations and virtual environments** for realistic gesture generation  
- **Transformations** (flips, rotations, lighting changes) to boost robustness  

---

## 🧠 Key Features
- 🧩 **Deep Learning Model (CNN)** for Arabic sign recognition  
- 📷 **Real-time Camera Detection** using OpenCV  
- ✋ **Dynamic ROI + Hand Detection** for stable predictions  
- 🧠 **Data Augmentation & GAN-based Synthesis** for higher accuracy  

---

## 🧰 Tech Stack
- **Languages:** Python  
- **Frameworks:** PyTorch, OpenCV, MediaPipe, Albumentations  
- **Tools:** Jupyter / Colab, GitHub, Git LFS  

---

## 🏁 Project Type
Developed as part of an applied **Computer Vision and Deep Learning** project for Arabic Sign Language recognition and real-time inference.
