# Import necessary libraries
from keras.models import Model
from keras.layers import Flatten,Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plot
from glob import glob

# Define image shape and dataset paths
IMAGESHAPE = [224, 224, 3] 

# Update these paths if needed (used for training and testing images)
training_data = r"C:\Users\aathi\OneDrive\Desktop\BATCH 15\archive\chest_xray\train"
testing_data = r"C:\Users\aathi\OneDrive\Desktop\BATCH 15\archive\chest_xray\test

# Load the pre-trained VGG16 model (excluding the top layers)
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)

# Freeze all layers so their weights won’t change during training
for each_layer in vgg_model.layers:
each_layer.trainable = False

# Get the number of classes (PNEUMONIA and NORMAL)
classes = glob(r"C:\Users\aathi\OneDrive\Desktop\BATCH 15\archive\chest_xray\train*") 

# Add flatten and dense layers for classification
flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)

# Build the complete model
final_model = Model(inputs=vgg_model.input, outputs=prediction) 
final_model.summary()

# Compile the model with loss function, optimizer, and metric
final_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

#Create data generators for training and testing (with augmentation)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
testing_datagen = ImageDataGenerator(rescale =1. / 255)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model architecture
final_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])

# Compile the model
final_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Data generators with rescaling
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
training_set = train_datagen.flow_from_directory(
    r'C:\Users\aathi\OneDrive\Desktop\BATCH 15\archive\chest_xray\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load test data
test_set = test_datagen.flow_from_directory(
    r'C:\Users\aathi\OneDrive\Desktop\BATCH 15\archive\chest_xray\test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Train the model
fitted_model = final_model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)
final_model.save('our_model.h5')
