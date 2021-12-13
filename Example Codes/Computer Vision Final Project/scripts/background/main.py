import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
import json
import glob
from sklearn.utils import resample
from keras.applications.imagenet_utils import preprocess_input
from tensorflow import keras
import sys
from scipy import optimize
from scipy.optimize import brenth


def create_labels_database(file):
    df = pd.read_csv(file, sep=" ", header=None)
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'spare', 0 : 'Label', 1 : 'Number'})
    df = df.drop(columns=['spare'])
    labels_list = df['Label'].tolist()
    return df, labels_list



# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]



def bisection_method(image, perturbations, eps, groundTruth_label, threshold):
	upper_bound = eps * 2
	lower_bound = eps
	while True:
		middle_bound = (lower_bound + upper_bound) / 2
		print(middle_bound)
		if (upper_bound - lower_bound) < threshold:
			break
		adv_x = image + middle_bound*perturbations
		adv_x = tf.clip_by_value(adv_x, -1, 1)
		output = get_imagenet_label(pretrained_model.predict(adv_x))
		pred_label = output[1]
		#print(pred_label)
		if pred_label == groundTruth_label:
			lower_bound = middle_bound
			print(middle_bound)
		if pred_label != groundTruth_label:
			upper_bound = middle_bound
			print(middle_bound)
	if pred_label == groundTruth_label:
		middle_bound = middle_bound + threshold
	epsilon = middle_bound
	print("Broken at : " + str(epsilon))
	return epsilon



def NN_pred(eps, image, label_index, cam_weights_normalized):
    label = tf.one_hot(label_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturbations = create_adversarial_pattern(image, label)
    adv_x = image + eps * perturbations * cam_weights_normalized
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    output = get_imagenet_label(pretrained_model.predict(adv_x))
    if output[1] == label_name:
        score = output[2]
    else:
        score = -abs(output[2])
    return score


def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad


# This function scales the mean to 1
def scaling(cam_weights):
	cam_weights_scaled = (cam_weights - np.min(cam_weights)) / (np.max(cam_weights) - np.min(cam_weights))
	cam_weights_normalized = cam_weights_scaled / np.mean(cam_weights_scaled)
	return cam_weights_normalized


# Grad-Cam Functions
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    #print("1 - grads here /n" + str(grads))
    #print("2 - grads shape /n" + str(grads.shape))



    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    #print("1 - shape here /n" + str(heatmap.shape))
    heatmap = tf.squeeze(heatmap)
    #print("2 - shape here /n" + str(heatmap.shape))


    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()



def save_and_display_gradcam(image_path, heatmap, alpha=1.0):
    # Load the original image
    img = keras.preprocessing.image.load_img(image_path)
    img = keras.preprocessing.image.img_to_array(img)
    #img = img.resize((224, 224))

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("binary")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((224, 224))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap #* alpha + img
    cam_weights = superimposed_img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    #superimposed_img.save(cam_path)

    # Display Grad CAM
    #display(Image(cam_path))

    return cam_weights





def write_to_file(file_path, data):
	file = open(file_path, "a")
	file.write(data + "\n")
	file.close()



pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
	weights='imagenet')
pretrained_model.trainable = False
# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions




df, labels_list = create_labels_database("imagenet_labels.txt")


#print(len(labels_list))


directory = "../../val_blurred"
folder_labels = os.listdir(directory) # dir is your directory path



#print(df.tail())
for subdir, dirs, files in os.walk(directory):
	image_label = subdir.split('/')[-1].lstrip().split(' ')[0]
	#print(image_label)
	for l in labels_list:
		if l == image_label:
			for file in files:
            #for file in resample(files, n_samples=5, replace=False, random_state=1):
				image_name = file
				image_path = os.path.join(subdir, file)
				image_raw = tf.io.read_file(image_path)
				image = tf.image.decode_image(image_raw)
				image = preprocess(image)
				try:
					output = get_imagenet_label(pretrained_model.predict(image))
					if output[0] == image_label:
						image_probs = pretrained_model.predict(image)
						loss_object = tf.keras.losses.CategoricalCrossentropy()

						# Get the input label of the image.
						label_row = df[df.eq(image_label).any(1)]
						label_index = label_row['Number'].values
						label_index = label_index[0]
						label_name = label_row[2].values
						label_name = label_name[0]

						label = tf.one_hot(label_index, image_probs.shape[-1])
						label = tf.reshape(label, (1, image_probs.shape[-1]))

						#print(label_name)
						perturbations = create_adversarial_pattern(image, label)

						# Time for Grad-Cam
						model_builder = keras.applications.MobileNetV2
						decode_predictions = keras.applications.mobilenet_v2.decode_predictions
						last_conv_layer_name = "Conv_1_bn"
						img_array = preprocess_input(get_img_array(image_path, size=(224, 224)))
						model = model_builder(weights="imagenet")
						# Remove last layer's softmax
						model.layers[-1].activation = None
						# Print what the top predicted class is
						#preds = model.predict(img_array)
						# Generate class activation heatmap
						heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
						cam_weights = save_and_display_gradcam(image_path=image_path, heatmap=heatmap)
						# Making flipping black to white - Object - 
						cam_weights_normalized = scaling(cam_weights)
						
						# Time for epsilon thresholding
						brentH_root = optimize.brenth(f=NN_pred, args=(image, label_index, cam_weights_normalized), a=0, b=1)

						data = "Image_label | " + image_label + " | Label Name | " + label_name + " | Epsilon Value | " + str(brentH_root) + " | File | " + image_path
						print(data)
						write_to_file("../../imagenet-val/background-pert.txt", data)


				except:
					data = "Problem | " + str(sys.exc_info()[0]) + " occured on image | " + image_path
					print(data)
					write_to_file("../../imagenet-val/background-pert.txt", data)