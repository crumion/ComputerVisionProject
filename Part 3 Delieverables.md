# Part 3 Delieverables


## Example

- A report (no page limit, but try to be concise; 3-4 pages should suffice) or a separate (from Parts 1 and 2) readme GitHub doc that includes:

- A short justification of the choice of classifier. For instance, if you selected SVM with RBF kernel, say why you think this classifier is good for your project. (3 points).

My goal is to evaluate the "robustness" or "trustworthiness" of a trained model or classifier. I decided to use a TensorFlow pretrained model, MobileNetV2, since it was used in many online examples. For my project, the classifier doesn't really matter, as I will be comparing one classifier to another. Since MobileNetV2 was trained on ImageNet, I was able to use a validation sample from Tiny-ImageNet, which features 200 classes at reduced image sizes (64, 64). The validation set had 10,000 images, 50 images per class. It also includes a text file mapping each file to an image label. I created a Pandas Dataframe to join these values with numeric and semantic meanings ('val_3.JPEG' : 'bathtub','n02808440', 884).


- A classification accuracy achieved on the training and validation sets. That is, how many objects were classified correctly and how many of them were classified incorrectly (It is better to provide a percentage instead of numbers). Students working on object detection may report Intersection over Union (Links to an external site.) averaged over the training and testing samples. More advanced students (especially those attending the 60535 section of the course) can select the performance metrics that best suit their given problem (for instance, Precision-Recall, f-measure, plot ROCs, etc.) and justify the use of the evaluation method. (3 points).

The model correctly classified 24.11% of the validation images. Some of the images were incorrectly formated which threw a value error, so I added exception handling to circumnavigate this error. 168 images were not tested and thrown out because of this error. Of the 10,000 images, 2411 remained correctly formatted, classified, and ready for adversarial attacks.

I used the Fast Gradient Sign Method to generate adversarial attacks on each image. I started at 0 epsilon and used step sizes of 0.00001 until the model predicted an incorrect classification. Since this was a linear search, my script took days to run only a fraction of the dataset. Instead of taking a smaller sample size, I decided to use a faster search algorithm. I wrote a bisectional method algorithm to find the smallest epsilon, which resulted in much faster speeds [1]. The method involved finding the midpoint of an upper and lower bound, and then testing which side of the bounds a failure occurs. Depending on the success or failure of the prediction, the lower or upper bound would become the middle bound, and a new midpoint would be computed. The algorithm would repeat until the difference between the upper and lower bounds was less than 0.00001 (step value). I was able to run the model over the entire validation set in only 6 hours. The epsilon values differed drastically, ranging from 0.000019 to 0.95. This was to be expected, because each image and classification is completely different. I averaged the epsilon values for each class, so I could access class robustness.

The most "sensitive" class was grasshopper, followed by fly and pizza (0.0000214440918, 0.000431060791, 0.0008377893066). The most "robust" classes were poncho, sock, and confectionery (0.46, 0.37, 0.31). These values tell us the average epsilon value for each class to fail (predict an incorrect classification). These show how far we have to travel in the direction of the gradient (signed method) to maximize the loss and fail the model. If we can easily fool a model, we should not readily trust it's ability to predict on new test sets.


- A short commentary related to the observed accuracy and ideas for improvements. For instance, if you see almost perfect accuracy on the training set, and way worse on the validation set, what does it mean? Is it good? If not, what do you think you could do to improve the generalization capabilities of your solution? (6 points)

I currently have a pipeline to test how robust each model and its classifiers are for Fast Gradient Sign Method. By generating these distributions, I can compare one model's robustness over another and gain confidence when showing these systems new images. These normal perturbations will be compared with the Grad-Cam Object perturbations and the Grad-Cam Background perturbations. We expect to see a range of average epsilon values across all three tests, which will add into our trusthworthiness metric. For example, if the Background perturbations are close to the Object perturbations, we can conclude the model isn't looking particularly at the object. However, if the average epsilon value is much higher for Background perturbations than for Object perturbations, we can conclude the model and class is more robust. There will be a failure point at each and every model, but the focus is to provide a range of confidence we can place on the model's ability to predict.

Initial trials of the Object and Background perturbations were inconclusive. Charts below show the insconsistencies between robustness, as we observed the opposite affect. Object perturbations were less sensitive to epsilon values, followed by Background and regular perturbations. We are still analyzing and interpreting these findings. These tests were run on a very small dataset (n=10), and on a very senstive class (bullfrog, which is in the top 1% of sensitive classes). Regardless of these findings, the relative pipelines are built and will be performed for the final sections of this project. However, we will be modifying our work going forward and using another way to interpret trustworthiness: heatmaps.

## Project Focus

Similarly to Grad-Cams, the gradient can also be visualized as a heatmap. This could be used as a replacement of Grad-Cams to estimate what the network is "looking" at. We wish to find a resemblance to how humans would classify objects. For example, we expect heatmaps to be concentrated around the face to classify a human. For future work, we will attempt to measure the compactness of a heatmap. The heatmaps below show the differences in concentration the network places on an object. We feel confident that a dense heatmap has higher robustness than one that is spread out, since there are likely only a few close features essential to classification. These concentrations can also be compared to Grad-Cams to see if the last convolutional layer agrees with the gradient (should be the same).


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


![5](https://user-images.githubusercontent.com/30506411/142565195-2f0d0ef4-2a7e-4464-a97c-e832ac8e6f51.png)


![6](https://user-images.githubusercontent.com/30506411/142565199-a8561c11-1b31-4016-9c8e-6de6609244d1.png)



![7](https://user-images.githubusercontent.com/30506411/142565208-bc8a8058-ef12-41c5-8960-15b494190590.png)


![8](https://user-images.githubusercontent.com/30506411/142565216-448c6adf-80ec-4a71-9d30-28f951f082df.png)



![9](https://user-images.githubusercontent.com/30506411/142565223-44af7634-5a41-436a-8ab9-d9e33973909f.png)




![10](https://user-images.githubusercontent.com/30506411/142565227-21cfe246-689a-4349-8766-e9b56fce964a.png)


![11](https://user-images.githubusercontent.com/30506411/142565237-312eec6a-8c8b-4bfc-8ca6-def26e2a1059.png)



## References
[1] https://en.wikipedia.org/wiki/Bisection_method



