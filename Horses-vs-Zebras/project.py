import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

########################################
#Loading:

image_size = (256, 256)
batch_size = 32

#Training dataset:
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"data",
	labels='inferred',
	validation_split=0.2,
	subset="training",
	seed=1337,
	image_size=image_size,
	batch_size=batch_size,
)

#Validation dataset:
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"data",
	labels='inferred',
	validation_split=0.2,
	subset="validation",
	seed=1337,
	image_size=image_size,
	batch_size=batch_size,
)

class_names = train_ds.class_names
print("CLASS NAMES: ", end = '')
print(class_names)

########################################
#Preprocessing:

#Data augmentation
data_augmentation = keras.Sequential(
	[
		layers.RandomFlip("horizontal"),
		layers.RandomRotation(0.1),
	]
)

augmented_train_ds = train_ds.map(
	lambda x, y: (data_augmentation(x, training=True), y))

#Performance: Use buffered prefetching to yield data from disk without bloacking I/O:
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

########################################
#Model:

def build_model(input_shape, num_classes):
	inputs = keras.Input(shape=input_shape)
	x = inputs
	
	#Rescaling:
	x = layers.Rescaling(1.0 / 255)(x)
	
	#Block 1:
	x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
	x = layers.MaxPooling2D(pool_size=(3, 3))(x)
	
	#Block 2:
	x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
	x = layers.MaxPooling2D(pool_size=(3, 3))(x)
	
	#Block 3:
	x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
	x = layers.MaxPooling2D(pool_size=(3, 3))(x)
	
	#Block 4:
	x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
	
	#Apply global average pooling to get flat feature vectors:
	x = layers.GlobalAveragePooling2D()(x)

	#Dropout:
	x = layers.Dropout(0.5)(x)
	
	#Dense classifier on bottom:
	outputs = layers.Dense(1, activation="sigmoid")(x)

	return keras.Model(inputs, outputs)

model = build_model(input_shape=image_size + (3,), num_classes=2)

model.summary()

########################################
#Training:

epochs = 2 #Avg epoch takes 40s, keep training time around 5 mins = 300 sec, 40 * 8 = 320 sec

model.compile(
	optimizer=keras.optimizers.Adam(1e-3),
	loss="binary_crossentropy",
	metrics=["accuracy"],
)

model.fit(
	train_ds, epochs=epochs, validation_data=val_ds,
)

########################################
#Inference/Prediction:

#Images not in test set:
horse_image_names = ["n02381460_20.jpg", "n02381460_440.jpg"]
zebra_image_names = ["n02391049_690.jpg", "n02391049_10910.jpg"]

img1 = keras.preprocessing.image.load_img(
	"testing/horses/n02381460_20.jpg", target_size=image_size
)
img2 = keras.preprocessing.image.load_img(
	"testing/horses/n02381460_440.jpg", target_size=image_size
)
img3 = keras.preprocessing.image.load_img(
	"testing/zebras/n02391049_690.jpg", target_size=image_size
)
img4 = keras.preprocessing.image.load_img(
	"testing/zebras/n02391049_10910.jpg", target_size=image_size
)

img_array1 = keras.preprocessing.image.img_to_array(img1)
img_array1 = tf.expand_dims(img_array1, 0)
img_array2 = keras.preprocessing.image.img_to_array(img2)
img_array2 = tf.expand_dims(img_array2, 0)
img_array3 = keras.preprocessing.image.img_to_array(img3)
img_array3 = tf.expand_dims(img_array3, 0) 
img_array4 = keras.preprocessing.image.img_to_array(img4)
img_array4 = tf.expand_dims(img_array4, 0)

predictions1 = model.predict(img_array1)
score1 = predictions1[0]
predictions2 = model.predict(img_array2)
score2 = predictions2[0]
predictions3 = model.predict(img_array3)
score3 = predictions3[0]
predictions4 = model.predict(img_array4)
score4 = predictions4[0]

scores = [score1, score2, score3, score4]

#Interpret/print results:
count = 0
for score in scores:
	if count < 2:
		print("Image: " + horse_image_names[count] +" (Known HORSE)")
	else:
		print("Image: " + zebra_image_names[count-2] + " (Known ZEBRA)")
	
	print(
		"This image is %.2f percent horse and %.2f percent zebra."
		% (100 * (1 - score), 100 * score))

	if score >= 0.5:
		prediction = 'zebra'
		probability = 1 - score
		print ("Probability = " + str(probability))
	else:
		prediction = 'horse'
		probability = score
		print ("Probability = " + str(probability))
	print("Prediction = " + prediction + "\n")
	count = count + 1