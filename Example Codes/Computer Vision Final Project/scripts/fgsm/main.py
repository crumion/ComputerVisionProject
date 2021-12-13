import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import glob
from sklearn.utils import resample
from scipy import optimize
from scipy.optimize import brenth
import sys



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
    print("Starting bisection method")
    upper_bound = eps * 2
    lower_bound = eps
    print("Upper bound : " + str(upper_bound))
    print("Lower bound : " + str(lower_bound))
    while True:
        print("Starting True loop")
        middle_bound = (lower_bound + upper_bound) / 2
        print(middle_bound)
        if (upper_bound - lower_bound) < threshold:
            print("Breaking out of the loop")
            break
        adv_x = image + middle_bound*perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        output = get_imagenet_label(pretrained_model.predict(adv_x))
        pred_label = output[1]
        #print(pred_label)
        if pred_label == groundTruth_label:
            print("BELOW break point")
            lower_bound = middle_bound
            print(middle_bound)
        if pred_label != groundTruth_label:
            print("ABOVE break point")
            upper_bound = middle_bound
            print(middle_bound)
    if pred_label == groundTruth_label:
        middle_bound = middle_bound + threshold
    epsilon = middle_bound
    print("Broken at : " + str(epsilon))
    return epsilon


def NN_pred(eps, image, label_index):
    label = tf.one_hot(label_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturbations = create_adversarial_pattern(image, label)
    adv_x = image + eps * perturbations
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
                        #print(label_name)

                        brentH_root = optimize.brenth(f=NN_pred, args=(image, label_index), a=0, b=1)
                        data = "Image_label | " + image_label + " | Label Name | " + label_name + " | Epsilon Value | " + str(brentH_root) + " | File | " + image_path
                        print(data)
                        write_to_file("../../imagenet-val/fgsm-pert.txt", data)


                except:
                    data = "Problem | " + str(sys.exc_info()[0]) + " occured on image | " + image_path
                    write_to_file("../../imagenet-val/gradpert.txt", data)
