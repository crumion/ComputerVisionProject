# Part 3 Delieverables


## Example

- A report (no page limit, but try to be concise; 3-4 pages should suffice) or a separate (from Parts 1 and 2) readme GitHub doc that includes:

- A short justification of the choice of classifier. For instance, if you selected SVM with RBF kernel, say why you think this classifier is good for your project. (3 points).

My goal is to evaluate the "robustness" or "trustworthiness" of a trained model or classifier. I decided to use a TensorFlow pretrained model, MobileNetV2, since it was used in many online examples. For my project, the classifier doesn't really matter, as I will be comparing one classifier to another. Since MobileNetV2 was trained on ImageNet, I was able to use a validation sample from Tiny-ImageNet, which features 200 classes at reduced image sizes (64, 64). The validation set had 10,000 images, 50 images per class. It also includes a text file mapping each file to an image label. I created a Pandas Dataframe to join these values with numeric and semantic meanings ('val_3.JPEG' : 'bathtub','n02808440', 884).


- A classification accuracy achieved on the training and validation sets. That is, how many objects were classified correctly and how many of them were classified incorrectly (It is better to provide a percentage instead of numbers). Students working on object detection may report Intersection over Union (Links to an external site.) averaged over the training and testing samples. More advanced students (especially those attending the 60535 section of the course) can select the performance metrics that best suit their given problem (for instance, Precision-Recall, f-measure, plot ROCs, etc.) and justify the use of the evaluation method. (3 points).

The model correctly classified 24.11% of the validation images. Some of the images were incorrectly formated which threw a value error, so I added exception handling to circumnavigate this error. 168 images were not tested and thrown out because of this error. Of the 10,000 images, 2411 remained correctly formatted, classified, and ready for adversarial attacks.


- A short commentary related to the observed accuracy and ideas for improvements. For instance, if you see almost perfect accuracy on the training set, and way worse on the validation set, what does it mean? Is it good? If not, what do you think you could do to improve the generalization capabilities of your solution? (6 points)

- Push your current codes realizing what you mention in the report to GitHub (3 points).

## Graphs

