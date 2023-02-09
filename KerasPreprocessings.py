"""
Copyright 2022 Mohammad Riazati
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import json
import os
from os import path
import sys
from keras.utils import np_utils
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import string #for lower, ...
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
from keras.utils import load_img
from keras.utils import img_to_array

output_location = "outputs" # "outputs/" folder in the current directory

#skip_data_generation = False
#dataset = "mnist"
#network = "lenet-quantized"

skip_data_generation = False
dataset = "mnist"
network = "lenet"

#skip_data_generation = True
#dataset = "imagenet"
#network = "ResNet50"

#skip_data_generation = False
#Note: Search SCALING
#dataset = "imagenet"
#network = "vgg16"

#dataset = "cifar10"
#dataset = "imagenet"

#network = "cifar10"
#network = "vgg"
#network = "alexnetcifar10"
#network = "alexnetimagenet"

#BEGIN: mnist and cifar10 DATASET<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
x_train = y_train = x_test = y_test = 0
if skip_data_generation == False and (dataset == "mnist" or dataset == "cifar10" or dataset == "imagenet"):
	if dataset == "mnist":
		############################Keras Dataset method
		from keras.datasets import mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

		############################Tensorflow dataset method
		if False: #to use this method, change it to true- This method is slower and creates some temprorary file.
			dataset_save_location = output_location
			data = tfds.builder("mnist", data_dir=dataset_save_location)
			data.download_and_prepare()
			num_classes = data.info.features['label'].num_classes
			Ntrain = data.info.splits['train'].num_examples
			Nvalidation = data.info.splits['test'].num_examples #for imagenet, it must be validation
			Nbatch = 32
			assert num_classes == 10
			assert Ntrain == 60000
			assert Nvalidation == 10000
		
			data = data.as_dataset()

			data_train, data_test = data["train"], data["test"]
			assert isinstance(data_train, tf.data.Dataset)
			assert isinstance(data_test, tf.data.Dataset)
			data_train = tfds.as_numpy(data_train)
			data_test = tfds.as_numpy(data_test)

			x_train = np.zeros((Ntrain, 28, 28, 1))
			y_train = np.zeros((Ntrain))
			x_test = np.zeros((Nvalidation, 28, 28, 1))
			y_test = np.zeros((Nvalidation))

			count = 0
			for example in data_train: 
				x_train[count], y_train[count] = example['image'], example['label']
				count += 1
			count = 0
			for example in data_test: 
				x_test[count], y_test[count] = example['image'], example['label']
				count += 1

	if dataset == "cifar10":
		assert false
		from keras.datasets import cifar10
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	if dataset == "imagenet":
		workspace = '.. ..\Datasets\imagenet\\'
		test_images_directory = workspace + 'tests-partial'
		#test_images_labels_file = workspace + 'tests-partial-labels.txt'
		classes_csv = workspace + 'classes.csv'

		#temp = open(test_images_labels_file).readlines()
		#test_images_labels = []
		#for element in temp: test_images_labels.append(element.strip())

		from keras.preprocessing.image import load_img
		from keras.preprocessing.image import img_to_array
		from keras.applications.vgg16 import preprocess_input
		from keras.applications.vgg16 import decode_predictions
		from keras.applications.vgg16 import VGG16		
		
		#all_images = np.zeros((len(test_images_labels), 224, 224, 3))
		all_images = np.zeros((100, 224, 224, 3))

		image_count = -1
		for filename in os.listdir(test_images_directory): #if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
			image_count += 1
			image_path = os.path.join(test_images_directory, filename)
			#print(image_path)
			
			image = load_img(image_path, target_size=(224, 224))
			# convert the image pixels to a numpy array
			image = img_to_array(image)
			#print(image.shape)

			image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
			#print(image.shape)
			image = preprocess_input(image)
			all_images[image_count] = image

#TF moved to bck.py

		#This model is to find the class for each image. The model in order to be converted is created later
		model = VGG16()
		model.summary()
	
		from numpy import genfromtxt
		imagenet_classes_from_csv = genfromtxt(classes_csv, delimiter=',', dtype=None)
		imagenet_classes = {}
		for i in imagenet_classes_from_csv: 
			imagenet_classes[i[0].decode()] = i[1]

		yhat = model.predict(all_images)
		print(np.shape(yhat))
		labels_temp = decode_predictions(yhat)
		print(np.shape(labels_temp))
		labels = []
		for i in labels_temp: 
			labels.append(imagenet_classes[i[0][0]] - 1) #-1 because VGG classes are from 1 to 1000, but LeNet (and DeepSimulator tool), begins from 0

		#x_test = np.zeros((len(test_images_labels), 224, 224, 3))
		#y_test = np.zeros((len(test_images_labels)))
		x_test = np.zeros((100, 224, 224, 3))
		y_test = np.zeros((100))

		x_test = np.array(all_images)
		y_test = np.array(labels)

		#count = 0
		#for example in data_test: 
		#	x_test[count], y_test[count] = image, test_images_labels[count]
		#	count += 1

		#tsdata = ImageDataGenerator()
		#data_test = tf.keras.preprocessing.image_dataset_from_directory(directory=test_images_directory, labels=test_images_labels, batch_size=1, image_size=(224,224), shuffle=False)

		#data_test = tfds.as_numpy(data_test)

		#x_test = np.zeros((len(test_images_labels), 224, 224, 3))
		#y_test = np.zeros((len(test_images_labels)))

		#count = 0
		#for example in data_test: 
		#	x_test[count], y_test[count] = example['image'], example['label']
		#	count += 1

	input_shape_size_x = x_test.shape[1] #e.g., for mnist, 28
	input_shape_size_y = x_test.shape[2] #e.g., for mnist, 28
	if len(x_test.shape) == 4:
		input_shape_size_z = x_test.shape[3]
	else:
		input_shape_size_z = 1
		
	if dataset != "imagenet":
		print(x_train.shape)
		print(y_train.shape)

	print(x_test.shape)
	print(y_test.shape)

	# Set numeric type to float32 from uint8 #for imagenet, from float64 to float32
	if dataset != "imagenet":
		x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	# Normalize value to [0, 1]
	if dataset != "imagenet" and network != "lenet-quantized":
		x_train /= 255
		x_test /= 255
	elif dataset == "mnist" and network == "lenet-quantized":
		x_train -= 128
		x_test -= 128
		#np.clip(x_train, -128, 127)
		#np.clip(x_test, -128, 127)
	#else: #Search SCALING
	#	#x_train /= 255
	#	x_test /= 255

	if dataset != "imagenet":
		data_type = "double"
	else:
		data_type = "float" #for mnist params were float and input was double, which I think was wrong!

	output_data_file = output_location + "/data.h"
	if not path.exists(output_data_file): #if data.h already exists, skips this part. If it is necessary to be performed, delete it (located in current directory)
		with open(output_data_file, 'w') as fout:
			#if input_shape_size_z == 1:
			#	print(data_type + " data[" + str(x_test.shape[0]) + '][' + str(x_test.shape[1]) + '][' + str(x_test.shape[2]) + '] =')
			#	fout.write(data_type + " data[" + str(x_test.shape[0]) + '][' + str(x_test.shape[1]) + '][' + str(x_test.shape[2]) + '] =\n')
			#else:
			#	print(data_type + " data[" + str(x_test.shape[0]) + '][' + str(x_test.shape[1]) + '][' + str(x_test.shape[2]) + '][' + str(x_test.shape[3]) + '] =')
			#	fout.write(data_type + " data[" + str(x_test.shape[0]) + '][' + str(x_test.shape[1]) + '][' + str(x_test.shape[2]) + '][' + str(x_test.shape[3]) + '] =\n')
			
			print(data_type + " data[" + str(x_test.shape[0]) + '][' + str(input_shape_size_x) + '][' + str(input_shape_size_y) + '][' + str(input_shape_size_z) + '] =')
			fout.write(data_type + " data[" + str(x_test.shape[0]) + '][' + str(input_shape_size_x) + '][' + str(input_shape_size_y) + '][' + str(input_shape_size_z) + '] =\n')

			i0 = 0
			i1 = 0
			i2 = 0
			progress = 1;
			#print("\t{", end="\n")
			fout.write("\t{\n")
			for i in x_test:
				if ((progress - 1) % 10 == 0 or progress == len(x_test)): print(str(progress) + "/" + str(len(x_test)))
				progress += 1

				#if i0 == 2: break
				if i0 != 0: 
					#print(",", end="\n")
					fout.write(",\n")
				i0 += 1
				i1 = i2 = 0
				#print("\t\t{", end="\n")
				fout.write("\t\t{\n")
				for j in i:
					if i1 != 0: 
						#print(",", end="\n")
						fout.write(",\n")
					i1 += 1
					i2 = 0
					#print("\t\t\t{", end="")
					fout.write("\t\t\t{")
					for k in j:
						if i2 != 0: 
							#print(",", end="")
							fout.write(",")
						i2 += 1
						if input_shape_size_z == 1:
							#print(k, end = "")
							fout.write("{")
							#fout.write(str(k))
							if dataset != "imagenet":
								fout.write('{0:.10f}'.format(k).rstrip('0').rstrip('.'))
							else:
								fout.write('{0:.10f}'.format(k[0]).rstrip('0').rstrip('.'))

							fout.write("}")
						else:
							i3 = 0
							fout.write("{")
							for l in k:
								if i3 != 0: 
									#print(",", end="")
									fout.write(",")
								i3 += 1
								#print(l, end = "")
								fout.write(str(l))
							fout.write("}")
					#print("}", end="")
					fout.write("}")
				#print("\n\t\t}", end="")
				fout.write("\n\t\t}")
			#print("\n\t};", end="")
			fout.write("\n\t};\n\n")

			#print("unsigned char labels[" + str(y_test.shape[0]) + '] =', end="")
			fout.write("unsigned char labels[" + str(y_test.shape[0]) + '] = ')

			i0 = 0
			#print("{", end="")
			fout.write("{")
			for i in y_test:
				#if i0 == 2: break
				if i0 != 0: 
					#print(",", end="")
					fout.write(",")
				i0 += 1
				if len(y_test.shape) == 1:
					#print(i, end = "")
					fout.write(str(i)) 
				else: #for cifar10, the labels are arrays of one element!
					#print(i[0], end = "")
					fout.write(str(i[0])) 
			#print("};\n\n", end="")
			fout.write("};\n\n")

	# Transform lables to one-hot encoding
	if dataset != "imagenet":
		y_train = np_utils.to_categorical(y_train, 10)
		y_test = np_utils.to_categorical(y_test, 10)

	# Reshape the dataset into 4D array
	if dataset != "imagenet":
		x_train = x_train.reshape(x_train.shape[0], input_shape_size_x, input_shape_size_y, input_shape_size_z)
		x_test = x_test.reshape(x_test.shape[0], input_shape_size_x, input_shape_size_y, input_shape_size_z) 

	if dataset != "imagenet":
		print(x_train.shape)
		print(y_train.shape)
	print(x_test.shape)
	print(y_test.shape)

#END: mnist and cifar10 and imagenet DATASET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

build_and_train = not path.exists(output_location + "/model.json")
#build_and_train = True

if build_and_train: #Model is not already generated and trained
	######################################################################
	#Define LeNet-5 Model
	from keras.models import Sequential
	from keras import models, layers
	from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
	import keras

	if network == "lenet":
		model = Sequential()
		model.add(layers.Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(28, 28, 1)))
		model.add(layers.MaxPool2D(strides=2))
		model.add(layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
		model.add(layers.MaxPool2D(strides=2))
		model.add(layers.Flatten())
		model.add(layers.Dense(120, activation='relu'))
		model.add(layers.Dense(84, activation='relu'))
		model.add(layers.Dense(10, activation='softmax'))
	######################### HAMID ############################################
	if network == "lenet-quantized":
		model = Sequential()
		model.add(layers.Conv2D(filters=6, kernel_size=(5,5),padding='same', activation='linear' ,input_shape=(28, 28, 1)))
		model.add(layers.MaxPool2D(strides=2))
		model.add(layers.Conv2D(filters=16,activation='linear', kernel_size=(5,5)))
		model.add(layers.MaxPool2D(strides=2))
		model.add(layers.Flatten())
		model.add(layers.Dense(120,activation='linear'))
		model.add(layers.Dense(84,activation='linear'))
		model.add(layers.Dense(10,activation='linear'))
	############################################################################
	
	if network == "cifar10":
		#https://keras.io/examples/cifar10_cnn/
		model = Sequential()
		model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
		model.add(Conv2D(32, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(10, activation='softmax'))

	if network == "vgg16":
		model = VGG16()

	if network == "ResNet50":
		model = tf.keras.applications.ResNet50(
			include_top=True,
			weights="imagenet",
			input_tensor=None
		)

	if network == "vgg-manual":
		#https://github.com/simongeek/KerasVGGcifar10/blob/master/vggkeras.py
		model = Sequential()
		model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3), name='block1_conv1'))
		model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

		model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
		model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

		model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

		model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

		model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
		model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4'))

		model.add(Flatten())

		model.add(Dense(4096, activation='relu'))
		model.add(Dense(4096, activation='relu', name='fc2'))
		model.add(Dense(10, activation='softmax'))

	if network == "alexnetcifar10":
		#https://github.com/night18/cifar-10-AlexNet/blob/master/alexnet.py
		model.add(layers.Conv2D(48, kernel_size=(3,3), activation='relu', padding='same', input_shape=x_train.shape[1:] ) )
		model.add(layers.MaxPool2D(strides=(2,2)))
		model.add(layers.Conv2D(96, kernel_size=(3,3), activation='relu', padding='same') )
		model.add(layers.MaxPool2D(strides=(2,2)) )
		model.add(layers.Conv2D(192, kernel_size=(3,3), activation='relu', padding='same') )
		model.add(layers.Conv2D(192, kernel_size=(3,3), activation='relu', padding='same') )
		model.add(layers.MaxPool2D(strides=(2,2)) )
		model.add(layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same') )
		model.add(layers.MaxPool2D(strides=(2,2)) )
		model.add(layers.Flatten())
		model.add(layers.Dense(512, activation='relu'))
		model.add(layers.Dense(256, activation='relu'))
		model.add(layers.Dense(10, activation='softmax'))


	from contextlib import redirect_stdout

	# Compile the model
	if network == "lenet" or network == "lenet-quantized" or network == "alexnetcifar10":
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=["accuracy"])
		print(model.summary())
		with open(output_location + '/modelsummary.txt', 'w') as f: 
			with redirect_stdout(f): 
				model.summary()

		if dataset == "mnist":
			batch_size = 128
		elif dataset == "cifar10":
			batch_size = 50
		elif dataset == "imagenet":
			print ("NOTTTTEEEEE")
			batch_size = 128

		epochs_param = 100
		if network == "lenet-quantized": 
			epochs_param = 1 #this accuracy doesn't matter. It is handeled in quantization process
			print ("Quantized network: Remember to copy ")

		hist = model.fit(x=x_train,y=y_train, epochs=epochs_param, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1) 
		#hist = model.fit(x=x_train,y=y_train, epochs=1, batch_size=batch_size, validation_data=(x_test, y_test), verbose=1)
	
	if network == "cifar10":
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])
		print(model.summary())
		with open(output_location + '/modelsummary.txt', 'w') as f: 
			with redirect_stdout(f): 
				model.summary()
		#hist = model.fit(x=x_train,y=y_train, epochs=1, batch_size=50, validation_data=(x_test, y_test), verbose=1)
		hist = model.fit(x=x_train,y=y_train, epochs=6, batch_size=50, validation_data=(x_test, y_test), verbose=1)

	if network == "vgg16" or network == "ResNet50":
		print(model.summary())
		with open(output_location + '/modelsummary.txt', 'w') as f: 
			with redirect_stdout(f): 
				model.summary()

	if network == "vgg-manual":
		from keras.optimizers import SGD
		sgd = SGD(lr=0.0005, decay=0, nesterov=True)
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics=["accuracy"])

		print(model.summary())
		with open(output_location + '/modelsummary.txt', 'w') as f: 
			with redirect_stdout(f): 
				model.summary()

		hist = model.fit(x=x_train,y=y_train, epochs=1, batch_size=50, validation_data=(x_test, y_test), verbose=1)
		#hist = model.fit(x=x_train,y=y_train, epochs=6, batch_size=50, validation_data=(x_test, y_test), verbose=1)

	#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
	#sudo pip install h5py
	from keras.models import model_from_json

	# serialize model to JSON
	model_json = model.to_json(indent=2)
	with open(output_location + "/model.json", "w") as json_file:
			json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(output_location + "/model.h5")
	print("Saved model to disk")

	#Download graphviz
	#https://graphviz.gitlab.io/_pages/Download/Download_windows.html
	#https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/5.0.0/windows_10_cmake_Release_graphviz-install-5.0.0-win64.exe
	#During installation, add Graphviz to path for all users

	#install pydot, pydotplus, graphviz, pydot-ng
	#restart VS

	import pydot
	import pydotplus
	import graphviz
	from keras.utils import plot_model
	model_img_file = output_location + "/model.png"
	tf.keras.utils.plot_model(model, to_file=model_img_file, 
                          show_shapes=True, 
                          show_layer_activations=True, 
                          show_dtype=True,
                          show_layer_names=True )

	#install netron library
	#https://github.com/lutzroeder/netron
	#https://www.youtube.com/watch?v=m9sfCvqH3Hw
	import netron
	netron.start(output_location + "/model.h5")
	

if network == "lenet-quantized": 
	print ("Quantized network: Remember to overwrite model.h5 with the quantized h5 (obtained from the quantization process)")

#https://github.com/pplonski/keras2cpp
#arch needed fro output arch
#model needed for weights file
from keras.models import model_from_json
arch = open(output_location + "/model.json").read()
model = model_from_json(arch)
model.load_weights(output_location + "/model.h5")

#model.compile(loss='categorical_crossentropy', optimizer='adadelta')
arch = json.loads(arch)

output_file = output_location + "/accuracy.txt"
if not path.exists(output_file) and not skip_data_generation:
	with open(output_file, 'w') as fout:
		import keras
		if network == "vgg-manual":
			from keras.optimizers import SGD
			sgd = SGD(lr=0.0005, decay=0, nesterov=True)
			model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics=["accuracy"])
			x_test = x_test[0:1000]
			y_test = y_test[0:1000]
		elif network == "vgg16":
			print("")
		elif network == "lenet" or network == "lenet-quantized" or network == "alexnetcifar10":
			model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=["accuracy"])		

		start_time = time.time()
		if network != "vgg16":
			test_score = model.evaluate(x_test, y_test)
		else:
			h5_from_another_source = True #This is when I received the quantized h5 and in this case the accuracy (which is for this specific input list and just compared with the previous result, not the real image class) might not be 100% 
			if (not h5_from_another_source):
				test_score = [0,1] #Assuming that accuracy is 100%. The result is only important when compared with the C version
			else:
				labels_source = labels 

				#This part is a copy of the previous time that the labels were generated
				yhat = model.predict(all_images)
				print(np.shape(yhat))
				labels_temp = decode_predictions(yhat)
				print(np.shape(labels_temp))
				labels = []
				for i in labels_temp: 
					labels.append(imagenet_classes[i[0][0]] - 1) #-1 because VGG classes are from 1 to 1000, but LeNet (and my GeneratedCPredict tool), begins from 0

				different_labels = 0
				labels_len = len(labels)
				for i in range(0, labels_len - 1):
					if (labels[i] != labels_source[i]):
					   different_labels = different_labels + 1

				test_score = [different_labels, (len(labels) - different_labels)/len(labels)] #Assuming that accuracy is 100%. The result is only important when compared with the C version

		print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100))
		fout.write("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100) + "\n")

		print("ExecTime: " + str(time.time() - start_time) + " seconds")
		fout.write("ExecTime: " + str(time.time() - start_time) + " seconds" + "\n")


output_arch_file = output_location + "/output_arch.py"
with open(output_arch_file, 'w') as fout:
	#print('layers ' + str(len(model.layers)) + '\n')

	current_layer_number = 0
	input_shape_x_from_input_layer = input_shape_y_from_input_layer = input_shape_y_from_input_layer = -1
	for ind, l in enumerate(arch["config"]["layers"]):
		current_layer_number = current_layer_number + 1
		#print("Current layer number is: ", current_layer_number)

		#print(ind, l)
		print('layer ' + str(ind) + ' ' + l['class_name'])

		if l['class_name'] == 'InputLayer':
			if 'batch_input_shape' in l['config']:
				input_shape_x_from_input_layer = l['config']['batch_input_shape'][1]
				input_shape_y_from_input_layer = l['config']['batch_input_shape'][2]
				input_shape_z_from_input_layer = l['config']['batch_input_shape'][3]
			else:
				assert false

		elif l['class_name'] == 'Conv2D':
			class_name = l['class_name']
			activation = l['config']['activation']
			input_shape_x = input_shape_y = input_shape_z = 0
			if 'batch_input_shape' in l['config']:
				input_shape_x = l['config']['batch_input_shape'][1]
				input_shape_y = l['config']['batch_input_shape'][2]
				input_shape_z = l['config']['batch_input_shape'][3]
			elif input_shape_x_from_input_layer >= 0 and ind <= 1:
				input_shape_x = input_shape_x_from_input_layer
				input_shape_y = input_shape_y_from_input_layer
				input_shape_z = input_shape_z_from_input_layer
				input_shape_x_from_input_layer = input_shape_y_from_input_layer = input_shape_y_from_input_layer = -1

			filters = l['config']['filters']
			kernel_size_x = l['config']['kernel_size'][0]
			kernel_size_y = l['config']['kernel_size'][1]

			padding = l['config']['padding']
			if padding == 'valid': padding = ''
			else: padding = ", padding='" + padding + "'"

			strides_x = l['config']['strides'][0]
			strides_y = l['config']['strides'][1]
			strides = ""
			if strides_x != 1 or strides_y != 1:
				strides = ", strides=(" + str(strides_x) + "," + str(strides_y) + ")"

			#model.add(Conv2D(filters=6, kernel_size=(5,5), padding='same', activation='relu', input_shape=(32, 32, 1)))
			keras_statement = "model.add(" + class_name + "(filters=" + str(filters) + ", kernel_size=(" + str(kernel_size_x) + "," + str(kernel_size_y) + ")" + padding + strides + ", activation='" + activation + "'"
			if input_shape_z:
				keras_statement += ", input_shape=(" + str(input_shape_x) + ", " + str(input_shape_y) + ", " + str(input_shape_z) + ")"
			keras_statement += "))"

			#print(keras_statement)
			fout.write(keras_statement + "\n")

		elif l['class_name'] == 'MaxPooling2D':
			class_name = "MaxPool2D" #l['class_name']
			strides_x = l['config']['strides'][0]
			strides_y = l['config']['strides'][1]

			#model.add(MaxPool2D(strides=2))
			keras_statement = "model.add(" + class_name + "(strides=(" + str(strides_x) + "," + str(strides_x) + ")))"

			#print(keras_statement)
			fout.write(keras_statement + "\n")

		elif l['class_name'] == 'Flatten':
			class_name = l['class_name']

			#model.add(Flatten())
			keras_statement = "model.add(Flatten())"

			#print(keras_statement)
			fout.write(keras_statement + "\n")

		elif l['class_name'] == 'Dense':
			class_name = l['class_name']
			activation = l['config']['activation']
			units = l['config']['units']

			#model.add(Dense(120, activation='relu'))
			keras_statement = "model.add(" + class_name + "(" + str(units) + ", activation='" + activation + "'))"

			#print(keras_statement)
			fout.write(keras_statement + "\n")
		else:
			class_name = l['class_name']

			#model.add(Dense(120, activation='relu'))
			keras_statement = "unsupported layer: class_name=" + class_name

			#print(keras_statement)
			fout.write(keras_statement + "\n")

#Search SCALING
scaling = 1
#if network == "vgg16":
#	scaling = 255

if network != "vgg16":
	data_type = "double"
else:
	data_type = "float"

output_file = output_location + "/param.h"
if not path.exists(output_file):
	with open(output_file, 'w') as fout:
		print('layers ' + str(len(model.layers)) + '\n')

		layers = []
		current_layer_number = 0
		for ind, l in enumerate(arch["config"]["layers"]):
			current_layer_number = current_layer_number + 1
			#print("Current layer number is: ", current_layer_number)

			#print(ind, l)
			print('layer ' + str(ind) + ' ' + l['class_name'] + '\n')

			#print(str(ind), l['class_name'])
			layers += [l['class_name']]
			if l['class_name'] == 'Conv2D' or l['class_name'] == 'Dense':
				#fout.write(str(l['config']['nb_filter']) + ' ' + str(l['config']['nb_col']) + ' ' + str(l['config']['nb_row']) + ' ')

				#if 'batch_input_shape' in l['config']:
				#    fout.write(str(l['config']['batch_input_shape'][1]) + ' ' + str(l['config']['batch_input_shape'][2]) + ' ' + str(l['config']['batch_input_shape'][3]))
				#fout.write('\n')

				if network == "lenet" or network == "lenet-quantized":
					temp = model.layers[ind-1].get_weights() #weights0 for layer 1, since layer zero is input layer #This worked once for lenet, now I try for VGG, doesn't work. Maybe a change in the TF system or the difference between VGG16 and LeNet
				else:
					temp = model.layers[ind].get_weights() 

				W = temp[0]
				print(W.shape)
			
				if(l['class_name'] == 'Conv2D'):
					#fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l['config']['border_mode'] + '\n')
					#print(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + '\n')

					#print(data_type + " weights_" + str(current_layer_number) + "[" + str(W.shape[0]) + '][' + str(W.shape[1]) + '][' + str(W.shape[2]) + '][' + str(W.shape[3]) + '] = {\n', end="")
					fout.write(data_type + " weights_" + str(current_layer_number-1) + "[" + str(W.shape[0]) + '][' + str(W.shape[1]) + '][' + str(W.shape[2]) + '][' + str(W.shape[3]) + '] = {\n')

					for i0 in range(W.shape[0]):
						if i0 != 0: 
							#print(",", end="\n")
							fout.write(",\n")
						#print("\t{", end="\n")
						fout.write("\t{\n")
						for i1 in range(W.shape[1]):
							if i1 != 0: 
								#print(",", end="\n")
								fout.write(",\n")
							#print("\t\t{", end="\n")
							fout.write("\t\t{\n")
							for i2 in range(W.shape[2]):
								if i2 != 0: 
									#print(",", end="\n")
									fout.write(",\n")
								#print("\t\t\t{", end="")
								fout.write("\t\t\t{")
								for i3 in range(W.shape[3]):
									if i3 != 0: 
										#print(",", end="")
										fout.write(",")
									#print(str(W[i2, i3, i0, i1]), end="")
									fout.write(str(W[i0, i1, i2, i3]))
								#print("}", end="")
								fout.write("}")
							#print("\n\t\t}", end="")
							fout.write("\n\t\t}")
						#print("\n\t}", end="")
						fout.write("\n\t}")
					#print("\n};", end="")
					fout.write("\n};")

				if(l['class_name'] == 'Dense'):
					#fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l['config']['border_mode'] + '\n')
					#print(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')

					#print(data_type + " weights_" + str(current_layer_number) + "[" + str(W.shape[0]) + '][' + str(W.shape[1]) + '] = {\n', end="")
					fout.write(data_type + " weights_" + str(current_layer_number-1) + "[" + str(W.shape[0]) + '][' + str(W.shape[1]) + '] = {\n')

					for i0 in range(W.shape[0]):
						if i0 != 0: 
							#print(",", end="\n")
							fout.write(",\n")
						#print("\t\t{", end="")
						fout.write("\t{")
						for i1 in range(W.shape[1]):
							if i1 != 0: 
								#print(",", end="")
								fout.write(",")
							##print(str(W[i0, i1]), end="")
							fout.write(str(W[i0, i1]))
						#print("}", end="")
						fout.write("}")
					#print("\n\t};", end="")
					fout.write("\n};")

				if network == "lenet" or network == "lenet-quantized":
					B = model.layers[ind-1].get_weights()[1] #weights0 for layer 1, since layer zero is input layer #This worked once for lenet, now I try for VGG, doesn't work. Maybe a change in the TF system or the difference between VGG16 and LeNet
				else:
					B = model.layers[ind].get_weights()[1]

				#print("\n\n", B.shape)

				#print("\n\n", B.shape)
				#print("\n\n\n" + data_type + " biases_" + str(current_layer_number) + "[" + str(B.shape[0]) + '] = ', end="")
				fout.write("\n\n\n" + data_type + " biases_" + str(current_layer_number-1) + "[" + str(B.shape[0]) + '] = ')
				#print("{", end="")
				fout.write("{")
				for i0 in range(B.shape[0]):
					if i0 != 0: 
						#print(",", end="")
						fout.write(",")
					#print(str(B[i0]), end="")
					fout.write(str(B[i0]/scaling)) #Search SCALING
				#print("}\n\n")
				fout.write("};\n\n")

print ("Done!")
