# Part 3 Delieverables


## Model and Classifier

My goal is to evaluate the "robustness" or "trustworthiness" of a trained model or classifier. I decided to use a TensorFlow pretrained model, MobileNetV2, since it was used in many online examples. For my project, the classifier doesn't really matter, as I will be comparing one classifier to another. Since MobileNetV2 was trained on ImageNet, I was able to use a validation sample from Tiny-ImageNet, which features 200 classes at reduced image sizes (64, 64). The validation set had 10,000 images, 50 images per class. It also includes a text file mapping each file to an image label. I created a Pandas Dataframe to join these values with numeric and semantic meanings ('val_3.JPEG' : 'bathtub','n02808440', 884).

## Accuracy and Performance

The model correctly classified 24.11% of the validation images. Some of the images were incorrectly formatted which threw a value error, so I added exception handling to circumnavigate this error. 168 images were not tested and thrown out because of this error. Of the 10,000 images, 2411 remained correctly formatted, classified, and ready for adversarial attacks.

I used the Fast Gradient Sign Method to generate adversarial attacks on each image. I started at 0 epsilon and used step sizes of 0.00001 until the model predicted an incorrect classification. Since this was a linear search, my script took days to run only a fraction of the data set (I have a CRC long script running it as of writing. My first script broke over an input ValueError, but I have committed the results I got up until that point). Instead of taking a smaller sample size, I decided to use a faster search algorithm. I wrote a bisectional method algorithm to find the smallest epsilon, which resulted in much faster speeds [1]. The method involved finding the midpoint of an upper and lower bound, and then testing which side of the bounds a failure occurs. Depending on the success or failure of the prediction, the lower or upper bound would become the middle bound, and a new midpoint would be computed. The algorithm would repeat until the difference between the upper and lower bounds was less than 0.00001 (step value). I was able to run the model over the entire validation set in only 6 hours. The epsilon values differed drastically, ranging from 0.000019 to 0.95. This was to be expected, because each image and classification is completely different. I averaged the epsilon values for each class, so I could access class robustness.

The most "sensitive" class was grasshopper, followed by fly and pizza (0.0000214440918, 0.000431060791, 0.0008377893066). The most "robust" classes were poncho, sock, and confectionery (0.46, 0.37, 0.31). These values tell us the average epsilon value for each class to fail (predict an incorrect classification). These show how far we have to travel in the direction of the gradient (signed method) to maximize the loss and fail the model. If we can easily fool a model, we should not readily trust it's ability to predict on new test sets.


## Room for Improvement

I currently have a pipeline to test how robust each model and its classifiers are for Fast Gradient Sign Method. By generating these distributions, I can compare one model's robustness over another and gain confidence when showing these systems new images. These normal perturbations will be compared with the Grad-Cam Object perturbations and the Grad-Cam Background perturbations. We expect to see a range of average epsilon values across all three tests, which will add into our trusthworthiness metric. For example, if the Background perturbations are close to the Object perturbations, we can conclude the model isn't looking particularly at the object. However, if the average epsilon value is much higher for Background perturbations than for Object perturbations, we can conclude the model and class is more robust. There will be a failure point at each and every model, but the focus is to provide a range of confidence we can place on the model's ability to predict.

Initial trials of the Object and Background perturbations were inconclusive. Charts below show the inconsistencies between robustness, as we observed the opposite affect. Object perturbations were less sensitive to epsilon values, followed by Background and regular perturbations. We are still analyzing and interpreting these findings. These tests were run on a very small data set (n=10), and on a very sensitive class (bullfrog, which is in the top 1% of sensitive classes). Regardless of these findings, the relative pipelines are built and will be performed for the final sections of this project. However, we will be modifying our work going forward and using another way to interpret trustworthiness: heatmaps.

Similarly to Grad-Cams, the gradient can also be visualized as a heatmap. This could be used as a replacement of Grad-Cams to estimate what the network is "looking" at. We wish to find a resemblance to how humans would classify objects. For example, we expect heatmaps to be concentrated around the face to classify a human. For future work, we will attempt to measure the compactness of a heatmap. The heatmaps of a bullfrog above show the differences in concentration the network places on an object (though these are Grad-Cams taken at different convolutional stages). We feel confident that a dense heatmap has higher robustness than one that is spread out, since there are likely only a few close features essential to classification. These concentrations can also be compared to Grad-Cams to see if the last convolutional layer agrees with the gradient (should be the same).

Another component of the pipeline that must be fixed before the final results is the code. The committed code is functional but not optimal. First, functions and classes need to be made and will improve code readability. Second, the code output to text files should be formatted better. I had to spend time data wrangling which could be prevented by better txt outputs. Third, the way the bisectional algorithm is written there is no check to see if the last prediction is equal to the first prediction. In some cases, the final epsilon value will not fully break the model. In these cases, the step value (0.00001) must be added to obtain the true breaking epsilon value. A simple logic statement after the loop will resolve this issue. I would also like to try other root finding methods to see if I can speed up the process. Upon review, the secant method looks promising to try, althrough there are a handful of potential methods [2]. Finally, the code does not stop adversarial attacks for incorrectly classified images. This will greatly improve the speed of the model and test only the necessary images. As always, the code could be threaded or optimized for GPU use. This will allow faster results and more scalable testing of the trustworthy pipeline.

## Relevant Code and Result Files

The relevant code has been uploaded to this GitHub repository.
- main-imagenet-200-8.py
This is the main python script responsible for the "fgsm_class_200_results.txt"

- fgsm_class_200_results.txt
The output of "main-imagenet-200-8.py". Note that these results do not measure confidence values associated with each epsilon value, which can be performed with a linear search if relevent later.

- broken_regular_class_flip_fgsm_adversarial_results.txt
This file reflects results from a previous script that did not have ValueError handling and was implementing a linear search instead of updated bisection method. Note these file contains incomplete results.

- broken_regular_fgsm_adversarial_results.txt
This file reflects the relevent classification flips per image with the previous code. Note it did not include error handling and contains incomplete results.


## Graphs


### Average Epsilon Failure by Each Class
![1](https://user-images.githubusercontent.com/30506411/142560750-b72dfbba-218e-4f89-a4f2-48b3094357cc.png)

### Average Epsilon Failure by Each Class (zoomed on uniform)
![2](https://user-images.githubusercontent.com/30506411/142560762-ad90ae3e-b2dd-49f9-a715-7e485c531e37.png)

### Top Most "Sensitive" Classes
![3](https://user-images.githubusercontent.com/30506411/142560770-523c6744-9436-44d8-8800-005a50ab804d.png)

### Top 10 Most "Robust" Classes
![4](https://user-images.githubusercontent.com/30506411/142560782-ef542084-c3fd-48b4-b387-2e4478cf44de.png)

### Regular Perturbations, Object, and Background: First Incorrect Classification
![5](https://user-images.githubusercontent.com/30506411/142565195-2f0d0ef4-2a7e-4464-a97c-e832ac8e6f51.png)

### Regular Perturbations: First Incorrect Classification
![6](https://user-images.githubusercontent.com/30506411/142565199-a8561c11-1b31-4016-9c8e-6de6609244d1.png)

### Background Pertubrations: First Incorrect Classification
![7](https://user-images.githubusercontent.com/30506411/142565208-bc8a8058-ef12-41c5-8960-15b494190590.png)

### Object Perturbations: First Incorrect Classification
![8](https://user-images.githubusercontent.com/30506411/142565216-448c6adf-80ec-4a71-9d30-28f951f082df.png)

### Object Perturbations: Changes in "Confidence"
![9](https://user-images.githubusercontent.com/30506411/142565223-44af7634-5a41-436a-8ab9-d9e33973909f.png)

### Regular Perturbations: Changes in "Confidence"
![10](https://user-images.githubusercontent.com/30506411/142565227-21cfe246-689a-4349-8766-e9b56fce964a.png)

### Background Perturbations: Changes in "Confidence"
![12](https://user-images.githubusercontent.com/30506411/142565296-4b16c580-0707-412d-9a76-fd7c92238c24.png)

### Regular Perturbation: First Incorrect Classification per Image
![11](https://user-images.githubusercontent.com/30506411/142565237-312eec6a-8c8b-4bfc-8ca6-def26e2a1059.png)


## References
[1] https://en.wikipedia.org/wiki/Bisection_method

[2] https://en.wikipedia.org/wiki/Root-finding_algorithms
