import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from keras.preprocessing import image as keras_image
from keras.applications.vgg16 import preprocess_input

# Load the trained model
model = load_model('our_model.h5')

# Chest X-ray file check
def is_chest_xray(filename):
    keywords = ['chest', 'xray', 'pneumonia', 'normal', 'person']
    return any(word in filename.lower() for word in keywords)

# Main window setup
root = tk.Tk()
root.title("Pneumonia Detector")
root.geometry("750x700")
root.configure(bg="#0b0c10")  # Dark blue-black background

# Styles
FONT = ('Segoe UI', 13)
RESULT_FONT = ('Segoe UI', 16, 'bold')

BTN_STYLE = {
    'font': FONT,
    'bg': '#0b3c5d',          # Dark blue button
    'fg': 'white',
    'activebackground': '#145374',  # Slightly lighter blue on hover
    'activeforeground': 'white',
    'relief': 'flat',
    'bd': 0,
    'padx': 15,
    'pady': 8
}

# Global path
selected_img_path = None

# Image display panel
img_panel = tk.Label(root, bg="#0b0c10")
img_panel.pack(pady=20)

# Result label
result_label = tk.Label(root, text="", font=RESULT_FONT, fg="#66fcf1", bg="#0b0c10")
result_label.pack(pady=10)

# Image selection function
def select_image():
    global selected_img_path
    result_label.config(text="")
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        if not is_chest_xray(file_path):
            messagebox.showerror("Unsupported File", "Please select a valid chest X-ray image.")
            return
    selected_img_path = file_path
    img = Image.open(file_path).resize((350, 350))
    img_tk = ImageTk.PhotoImage(img)
    img_panel.config(image=img_tk)
    img_panel.image = img_tk

# Prediction function
def predict():
    if not selected_img_path:
        messagebox.showwarning("No Image", "Please upload an image first.")
        return
    try:
        img = keras_image.load_img(selected_img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)

        if prediction.shape[1] == 1:
            result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
        else:
            result = "Normal" if prediction[0][0] > prediction[0][1] else "Pneumonia"

        result_label.config(text=result)

    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error: {str(e)}")

# Buttons
upload_btn = tk.Button(root, text="Upload Chest X-ray", command=select_image, **BTN_STYLE)
upload_btn.pack(pady=10)

predict_btn = tk.Button(root, text="Predict", command=predict, **BTN_STYLE)
predict_btn.pack(pady=5)

# Footer
footer = tk.Label(root, text="Powered by Keras + VGG16", font=('Segoe UI', 10), fg="#c5c6c7", bg="#0b0c10")
footer.pack(side="bottom", pady=15)

root.mainloop()
