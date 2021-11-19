import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json



def display_images(image, description):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()



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




pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions



#image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')


df = pd.read_csv("imagenet_labels.txt", sep=" ", header=None)
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'spare', 0 : 'Label', 1 : 'Number'})
df = df.drop(columns=['spare'])

val_labels = pd.read_csv('val_annotations.txt', sep="\t", header=None)
print(val_labels.head())
val_labels.columns = ["File", "Label", "a", "b", "c", "d"]
val_labels = val_labels.drop(columns=['a', 'b', 'c', 'd'])


database = pd.merge(left=val_labels, right=df, left_on='Label', right_on='Label')
database = database.rename(columns={2: 'Word'})

labels_df = database.drop_duplicates(subset=['Label'])
labels_list = labels_df['Label'].tolist()

directory = "../../../tiny-imagenet-200/val/images"
folder = os.listdir(directory) # dir is your directory path



for l in labels_list:
  rslt_df = database.loc[database["Label"] == l]
  #print(rslt_df.sample(frac=0.5, replace=False, random_state=1))
  for im in rslt_df.iterrows():
    row_series = im[1].values
    image = row_series[0]
    img = image
    image_path = directory + "/" + str(image)
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)
    image = preprocess(image)
    try:
      image_probs = pretrained_model.predict(image)
      loss_object = tf.keras.losses.CategoricalCrossentropy()

      image_row = database[database.eq(img).any(1)]
      label = int(image_row['Number'].item())
      #print(label)
      img_path = directory + "/" + str(img)


        # Get the input label of the image.
      label_index = label
      label = tf.one_hot(label_index, image_probs.shape[-1])
      label = tf.reshape(label, (1, image_probs.shape[-1]))

      #ep = np.arange(0.0, 1.0, 0.00001)

        #epsilons = list(ep)
      #descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]

      perturbations = create_adversarial_pattern(image, label)

      adv_x = image
      adv_x = tf.clip_by_value(adv_x, -1, 1)
      output = get_imagenet_label(pretrained_model.predict(adv_x))
      groundTruth_label = output[1]
      print(groundTruth_label)
      pred_label = " "
      eps = 2.0
      while groundTruth_label != pred_label:
        eps = eps / 2
        adv_x = image + eps*perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        output = get_imagenet_label(pretrained_model.predict(adv_x))
        pred_label = output[1]
      #print(eps)
      upper_bound = eps * 2
      lower_bound = eps
      while True:
        middle_bound = (lower_bound + upper_bound) / 2
        if (upper_bound - lower_bound) < 0.00001:
          break
        #print(str(img) + ": " + str(eps))
        adv_x = image + middle_bound*perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        output = get_imagenet_label(pretrained_model.predict(adv_x))
        pred_label = output[1]
        print(pred_label)
        if pred_label == groundTruth_label:
          lower_bound = middle_bound
          print(middle_bound)
        if pred_label != groundTruth_label:
          upper_bound = middle_bound
          print(middle_bound)
      print("Flipped classes at " + str(middle_bound))
      failure = "File:" + str(img) + " Class: " + str(groundTruth_label) + " Failured class: " + str(pred_label) + " Eps: " + str(middle_bound)
      print(failure)
      file_path = "../../imagenet-200-results/fgsm_class_200_sampled.txt"
      file = open(file_path, "a")
      file.write(failure + "\n")
      file.close()
    except ValueError:
      write_to_file = image_path + " was skipped due to a ValueError."
      file_path = "../../imagenet-200-results/fgsm_class_200_sampled.txt"
      file = open(file_path, "a")
      file.write(write_to_file + "\n")
      file.close()
