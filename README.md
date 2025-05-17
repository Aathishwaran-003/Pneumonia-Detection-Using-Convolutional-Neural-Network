# 🫁 Pneumonia Detection Using Convolutional Neural Networks (CNN)

This project implements a deep learning-based solution for detecting pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) model. It leverages TensorFlow and Keras to train a binary classification model that distinguishes between normal and pneumonia-infected lungs.

# 📌 Project Highlights

📷 Medical Imaging: Uses chest X-ray images as input data.
🧠 Deep Learning Model: Implements CNN using Keras for feature extraction and classification.
🔍 Preprocessing: Rescaling and augmenting data using ImageDataGenerator.
📈 Binary Classification: Distinguishes between normal and pneumonia images.
💾 Model Saving: The trained model is saved as our_model.h5 for reuse.
🖼️ GUI Integration: A basic Tkinter interface is integrated to allow users to upload an image and get predictions.


# 🧰 Tech Stack
## 💻 Software & Libraries

- Python
- TensorFlow / Keras
- OpenCV (optional)
- Tkinter (for GUI)
- NumPy
- PIL (Python Imaging Library)

## 🧪 Model Architecture

- Conv2D and MaxPooling layers
- Flattening and Dense layers
- Final sigmoid activation for binary output


# 📁 Folder Structure

pneumonia-detection/
│
├── chest_xray/              # Dataset (train/test/val)
├── pneumonia detection using CNN.ipynb
├── our_model.h5             # Saved trained model
├── gui.py                   # Tkinter-based GUI for testing (if separated)
├── README.md
└── requirements.txt


# 📦 Installation & Setup

## 1. Clone the Repository:
```bash
   git clone https://github.com/your-username/pneumonia-detection.git
   cd pneumonia-detection
```
   
## 2. Install Required Packages:
```bash
    pip install -r requirements.txt
```
   
## Or manually:
   ```bash
  pip install tensorflow keras numpy pillow opencv-python
   ```
## 3. Dataset Setup:
Download and extract the dataset from Kaggle Chest X-ray dataset and place it in the chest_xray/ folder.

## 4. Train the Model:
Run the notebook or script:
  ```bash
   python pneumonia_detection_cnn.py
```
   
## 5. Launch the GUI (Optional):
```bash
   python gui.py
```

   
   
# 🧪 Dataset Details

- Source: Kaggle (Chest X-Ray Images)
- Categories: PNEUMONIA and NORMAL
- Directory Structure:
chest_xray/
├── train/
├── test/
└── val/


# 🧠 Model Performance

- ✅ Binary classification (sigmoid)
- 📊 Evaluated using accuracy during training/validation
- 📁 Model saved as: our_model.h5


# 🚀 Future Improvements

- Integration of VGG16, ResNet, or other pre-trained models for better accuracy
- Deploy the model as a web app using Flask or Streamlit
- Add Grad-CAM visualization for model explainability
- Incorporate multi-class classification with more lung diseases


 
