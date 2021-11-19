# Part 3 Delieverables


## Example

- A report (no page limit, but try to be concise; 3-4 pages should suffice) or a separate (from Parts 1 and 2) readme GitHub doc that includes:

- A short justification of the choice of classifier. For instance, if you selected SVM with RBF kernel, say why you think this classifier is good for your project. (3 points).

My goal is to evaluate the "robustness" or "trustworthiness" of a trained model or classifier. I decided to use a TensorFlow pretrained model, MobileNetV2, since it was used in many online examples. For my project, the classifier doesn't really matter, as I will be comparing one classifier to another. Since MobileNetV2 was trained on ImageNet, I was able to use a validation sample from Tiny-ImageNet, which features 200 classes at reduced image sizes (64, 64). The validation set had 10,000 images, 50 images per class. It also includes a text file mapping each file to an image label. I created a Pandas Dataframe to join these values with numeric and semantic meanings ('val_3.JPEG' : 'bathtub','n02808440', 884).


- A classification accuracy achieved on the training and validation sets. That is, how many objects were classified correctly and how many of them were classified incorrectly (It is better to provide a percentage instead of numbers). Students working on object detection may report Intersection over Union (Links to an external site.) averaged over the training and testing samples. More advanced students (especially those attending the 60535 section of the course) can select the performance metrics that best suit their given problem (for instance, Precision-Recall, f-measure, plot ROCs, etc.) and justify the use of the evaluation method. (3 points).

The model correctly classified 24.11% of the validation images. Some of the images were incorrectly formated which threw a value error, so I added exception handling to circumnavigate this error. 168 images were not tested and thrown out because of this error. Of the 10,000 images, 2411 remained correctly formatted, classified, and ready for adversarial attacks.

I used the Fast Gradient Sign Method to generate adversarial attacks on each image. I started at 0 epsilon and used step sizes of 0.00001 until the model predicted an incorrect classification. Since this was a linear search, my script took days to run only a fraction of the dataset. Instead of taking a smaller sample size, I decided to use a faster search algorithm. I wrote a bisectional method algorithm to find the smallest epsilon, which resulted in much faster speeds. The method involved finding the midpoint of an upper and lower bound, and then testing which side of the bounds a failure occurs. Depending on the success or failure of the prediction, the lower or upper bound would become the middle bound, and a new midpoint would be computed.
I was able to run the model I initially had used a linear search algorithm to find where the classification


- A short commentary related to the observed accuracy and ideas for improvements. For instance, if you see almost perfect accuracy on the training set, and way worse on the validation set, what does it mean? Is it good? If not, what do you think you could do to improve the generalization capabilities of your solution? (6 points)

- Push your current codes realizing what you mention in the report to GitHub (3 points).


## Graphs


### Average Epsilon Failure by Each Class
![1](https://user-images.githubusercontent.com/30506411/142560750-b72dfbba-218e-4f89-a4f2-48b3094357cc.png)


### Average Epsilon Failure by Each Class (zoomed on uniform)
![2](https://user-images.githubusercontent.com/30506411/142560762-ad90ae3e-b2dd-49f9-a715-7e485c531e37.png)



### Top Most "Sensitive" Classes
![3](https://user-images.githubusercontent.com/30506411/142560770-523c6744-9436-44d8-8800-005a50ab804d.png)



### Top 10 Most "Robust" Classes
![4](https://user-images.githubusercontent.com/30506411/142560782-ef542084-c3fd-48b4-b387-2e4478cf44de.png)







