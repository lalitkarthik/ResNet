import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os

train_images = np.load('train_images.npy')
test_images = np.load('test_images.npy')
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ["tf_gpu_allocator"]="cuda_malloc_async"

physical_devices=tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and configured.")
else:
    print("No GPU found.")

with open('model_config.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

model.load_weights('model_weights.h5')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

batch_size = 8
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

model.fit(train_dataset, epochs=10, validation_data=test_dataset, verbose=1)

model.save('alphabet_trained_model.h5')

print("Model trained and saved successfully.")
