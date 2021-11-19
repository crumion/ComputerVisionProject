# Computer Vision 1 Semester Project

This document entails the Computer Vision 1 Semester Project by Colton Crum, heavily led, helped, and encouraged by Dr. Adam Czajka.

Adapted work from an ongoing research project through the Computer Vision Research Lab, University of Notre Dame.

# Part 1 Delieverables

## Dataset

I will be using ImageNet for my dataset. ImageNet contains over 14 million annotated images across over 20,000 categories. Many proposed models for my project are trained on ImageNet. ImageNet has built arguments to split the dataset into training, validation, and testing sets, though these will likely not be as relevant to my project. I have already pulled the dataset onto my local machine for previous work [5].

Due to the rapid tests of the project, I have used a smaller version of ImageNet, Tiny-ImageNet, which contains 200 classes and reduced file sizes of (64, 64).

## Proposed Solution

I will be modifying some of the work I've been doing on the Crane research project, specifically within the context of trustworthiness in AI. Trustworthy AI involves frameworks, metrics, or tools that help us trust the conclusions of deep learning models. These techniques help break down "black boxes" and make models more transparent, more graceful to failures, and robust. In a world where models are being used in high stakes situations and lives may be at stake, it's essential that we build trustworthy and explainable features into AI so that we can effectively work in unison on more expansive projects in the future.

For my semester project, I will be forming a reliability metric to access the models based on random, out of class inputs. My previous work has involved adding random perturbations onto different parts of an image, and observing the differences in model confidence. Using object masking, I have added perturbations to the background, object, and entire image to compare confidence scores and measure different ways the model classify an image. I have used VGG16, VGG19, and Xception and evaluated their robustness to these adversarial attacks. For my semester project, I plan to use these models in more diverse ways, eventually creating a set of metrics to evaluate their performance and the various ways we can access their trustworthiness. In theory, by checking their responses to different situations (perturbations, rotations, etc.) we can formulate a trust score, and thus we can have a better measure in their ability to perform tasks accurately than merely observing confidence scores.

# Part 2 Delieverables

## Methods Applied for data pre-processing and feature extraction

After downloading the images, I had to rename files to reflect the image content. I set up local directories so I could iterate over them easily. I also fixed file extensions, renaming all extensions to reflect .jpg (some were jpeg).

After some initial data preparations, I started to run some preliminary models. More particularly, I tested my images using MobileNetV2 to observe some baseline predictions. Some images were incorrectly classified, so I removed them from the dataset. I chose to use MobileNetV2 because it fit well with existing tutorials. My project isn't specific to certain architectures or datasets, so these choices are somewhat arbitrary.

After running some of the images, I ran some adversarial attacks (FGSM) and observed how the model reacted. Consequently, I used Grad-CAM masking to focus the perturbations onto the last convolutional layer. Explanations and details in the following section.

## Justification of Methods

### Softmax Scores ≠ Probability

I decided to start with MobileNetV2 in evaluating its own trustworthiness. For my project, I'm looking for more explainability than merely a "confidence" score. These confidence scores are simply softmax scores, which are often described as probabilities. However, the softmax score is not a probability, so we must be more sophisticated in the ways we evaluate our models. When running these "black box" models, we can't always trust what we see.

### Image Alterations

Since softmax scores don't necessarily reflect the model's prediction probability, we need to find other ways of evaluating the model. One approach is to progressively alter the image until the model incorrectly classifies the image. This shows model robustness, and gives us a level of trust in the system as a whole. These tests would need to be performed on a very large dataset, and on every class. For example, some classes may be more resistent to changes than others. Ultimately, I would like to gather a smooth distribution of different alterations over a number of simulations so that a trustworthy metric could be calculated per model, per class.

### Perturbations

One of the first common adversarial attacks on nueral networks uses Fast Gradient Signed Method (FGSM), first proposed by Goodfellow et al. FGSM uses the gradients of the loss function with respect to the image [2]. These perturbations are one way to fool the model, and observe how quickly it breaks by adjusting the epsilon. If the model is not fooled until high levels of epsilon, then it is trustworthy. If the model is fooled rather quickly, we cannot be confident in its deployment on test data. Below is some details regarding the perturbation formula:

<img width="450" alt="Screen Shot 2021-10-14 at 1 55 57 PM" src="https://user-images.githubusercontent.com/30506411/137370941-35ba6fbc-56a5-4b86-b1b6-a5c4ce5a4ba1.png">
Source: [6]

These perturbations can be focused to different parts of the image using other methods like Grad-CAM.

### Grad-CAM Heatmaps, Perturbations

Grad-CAMs have become a popular way to evaluate what each model is "looking" at. Grad-CAMs generate heatmaps using the last convolutional layer. These heatmaps can provide practitioners a visual that describes where the model is focusing its attention at the moment before it makes its prediction [1]. I would like to use Grad-CAMs to focus perturbations to where the model is looking at. At this step, we should see the model less tolerant to changes and more easily flipped to incorrect classifications. Next, I would like to flip the Grad-CAM to focus only on the background of the image. We expect to see the model more tolerate to perturbations than the Grad-CAM object perturbations, and the general perturbations. If we repeat these steps with many samples from each class, we should get a smooth distribution of general perturbations, Grad-CAM object-focused perturbations, and Grad-CAM background-focused perturbations. By using these framework, we can apply mathematical operations to get a trustworthy score. This score determines the tolerance for each trained models's classes. For example, some classes may have extremely high confidence (> 90%), but with a slight change to the image it will flip to an incorrect classification. Therefore, the confidence score is misleading, but our trustworthy metric will capture the model's sensitivity to these changes and give us appropiate model trust.


## A few illustrations demonstrating how your methods processed training data

## Preliminary Results
<img width="600" alt="perturbations_graph" src="https://user-images.githubusercontent.com/30506411/137355770-3873f5a2-1cc1-4f83-a046-dca35aa28c76.png">

This graph shows how the model's confidence scores changes at different epsilon levels. Perhaps the most intriguing part is the model gets more "confident" with its incorrect classifications. Ultimately, this graphic shows that the confidence scores are not a reliable metric. 

## Grad-CAM Heatmaps

### First Convolutional Layer
<img width="468" alt="bullfrog1" src="https://user-images.githubusercontent.com/30506411/137356367-55cba57f-8d8c-42b6-8f56-8bf0706effd9.png">

### Hand Picked in Between Layers
<img width="468" alt="bullfrog2" src="https://user-images.githubusercontent.com/30506411/137356404-56d1e07d-e633-4682-8c2a-97d933dff98a.png">

<img width="468" alt="bullfrog3" src="https://user-images.githubusercontent.com/30506411/137356423-0ad0e49d-93c8-4477-8287-166d267b8d39.png">

### Last Convolutional Layer
<img width="468" alt="bullfrog4" src="https://user-images.githubusercontent.com/30506411/137356445-b2f51f14-5a61-4f7b-b243-5e0e324499dc.png">

This heatmap is used to increase perturbation intensity around the object.

<img width="600" alt="Picture1" src="https://user-images.githubusercontent.com/30506411/137511787-9bc51c3e-4861-4d88-9900-4abe1b18fc63.png">

Some preliminary results on a single image.

# Part 3 Delieverables

## Model and Classifier

My goal is to evaluate the "robustness" or "trustworthiness" of a trained model or classifier. I decided to use a TensorFlow pretrained model, MobileNetV2, since it was used in many online examples. For my project, the classifier doesn't really matter, as I will be comparing one classifier to another. Since MobileNetV2 was trained on ImageNet, I was able to use a validation sample from Tiny-ImageNet, which features 200 classes at reduced image sizes (64, 64). The validation set had 10,000 images, 50 images per class. It also includes a text file mapping each file to an image label. I created a Pandas Dataframe to join these values with numeric and semantic meanings ('val_3.JPEG' : 'bathtub','n02808440', 884).

## Accuracy and Performance

The model correctly classified 24.11% of the validation images. Some of the images were incorrectly formated which threw a value error, so I added exception handling to circumnavigate this error. 168 images were not tested and thrown out because of this error. Of the 10,000 images, 2411 remained correctly formatted, classified, and ready for adversarial attacks.

I used the Fast Gradient Sign Method to generate adversarial attacks on each image. I started at 0 epsilon and used step sizes of 0.00001 until the model predicted an incorrect classification. Since this was a linear search, my script took days to run only a fraction of the dataset (I have a CRC long script running it as of writing. My first script broke over an input ValueError, but I have committed the results I got up until that point). Instead of taking a smaller sample size, I decided to use a faster search algorithm. I wrote a bisectional method algorithm to find the smallest epsilon, which resulted in much faster speeds [3]. The method involved finding the midpoint of an upper and lower bound, and then testing which side of the bounds a failure occurs. Depending on the success or failure of the prediction, the lower or upper bound would become the middle bound, and a new midpoint would be computed. The algorithm would repeat until the difference between the upper and lower bounds was less than 0.00001 (step value). I was able to run the model over the entire validation set in only 6 hours. The epsilon values differed drastically, ranging from 0.000019 to 0.95. This was to be expected, because each image and classification is completely different. I averaged the epsilon values for each class, so I could access class robustness.

The most "sensitive" class was grasshopper, followed by fly and pizza (0.0000214440918, 0.000431060791, 0.0008377893066). The most "robust" classes were poncho, sock, and confectionery (0.46, 0.37, 0.31). These values tell us the average epsilon value for each class to fail (predict an incorrect classification). These show how far we have to travel in the direction of the gradient (signed method) to maximize the loss and fail the model. If we can easily fool a model, we should not readily trust it's ability to predict on new test sets.


## Room for Improvement

I currently have a pipeline to test how robust each model and its classifiers are for Fast Gradient Sign Method. By generating these distributions, I can compare one model's robustness over another and gain confidence when showing these systems new images. These normal perturbations will be compared with the Grad-Cam Object perturbations and the Grad-Cam Background perturbations. We expect to see a range of average epsilon values across all three tests, which will add into our trusthworthiness metric. For example, if the Background perturbations are close to the Object perturbations, we can conclude the model isn't looking particularly at the object. However, if the average epsilon value is much higher for Background perturbations than for Object perturbations, we can conclude the model and class is more robust. There will be a failure point at each and every model, but the focus is to provide a range of confidence we can place on the model's ability to predict.

Initial trials of the Object and Background perturbations were inconclusive. Charts below show the insconsistencies between robustness, as we observed the opposite affect. Object perturbations were less sensitive to epsilon values, followed by Background and regular perturbations. We are still analyzing and interpreting these findings. These tests were run on a very small dataset (n=10), and on a very senstive class (bullfrog, which is in the top 1% of sensitive classes). Regardless of these findings, the relative pipelines are built and will be performed for the final sections of this project. However, we will be modifying our work going forward and using another way to interpret trustworthiness: heatmaps.

Similarly to Grad-Cams, the gradient can also be visualized as a heatmap. This could be used as a replacement of Grad-Cams to estimate what the network is "looking" at. We wish to find a resemblance to how humans would classify objects. For example, we expect heatmaps to be concentrated around the face to classify a human. For future work, we will attempt to measure the compactness of a heatmap. The heatmaps below show the differences in concentration the network places on an object. We feel confident that a dense heatmap has higher robustness than one that is spread out, since there are likely only a few close features essential to classification. These concentrations can also be compared to Grad-Cams to see if the last convolutional layer agrees with the gradient (should be the same).

Another component of the pipeline that must be fixed before the final results is the code. The committed code is functional but not optimal. First, functions and classes need to be made and will improve code readability. Second, the code output to text files should be formatted better. I had to spend time data wrangling which could be prevented by better txt outputs. Third, the way the bisectional algorithm is written there is no check to see if the last prediction is equal to the first prediction. In some cases, the final epsilon value will not fully break the model. In these cases, the step value (0.00001) must be added to obtain the true breaking epsilon value. A simple logic statement after the loop will resolve this issue. I would also like to try other root finding methods to see if I can speed up the process. Upon review, the secant method looks promising to try, althrough there are a handfull of potential methods [4]. Finally, the code does not stop adversarial attacks for incorrectly classified images. This will greatly improve the speed of the model and test only the necessary images. As always, the code could be threaded or optimized for GPU use. This will allow faster results and more scalable testing of the trustwothy pipeline.

## Relevant Code and Result Files

The relevent code has been uploaded to this GitHub repository.
- main-imagenet-200-8.py
This is the main python script responsible for the "fgsm_class_200_results.txt"

- fgsm_class_200_results.txt
The output of "main-imagenet-200-8.py". Note that these results do not measure confidence values associated with each epsilon value, which can be performed with a linear search if relevent later.

- broken_regular_class_flip_fgsm_adversarial_results.txt
This file reflects results from a previous script that did not have ValueError handling and was implementing a linear search instead of updated bisection method. Note these file contains incomplete results.

- broken_regular_fgsm_adversarial_results.txt
This file reflects the relevent classification flips per image with the previous code. Note it did not include error handling and contains incomplete results.

## Graphics


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

## Acknowledgment
This work is an adaptation of an ongoing research project through the Computer Vision Research Lab at the University of Notre Dame, advised by Dr. Adam Czajka.

## References
[1]	R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” Int. J. Comput. Vis., vol. 128, no. 2, pp. 336–359, Feb. 2020, doi: 10.1007/s11263-019-01228-7.

[2]	I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and Harnessing Adversarial Examples,” ArXiv14126572 Cs Stat, Mar. 2015, Accessed: Oct. 14, 2021. [Online]. Available: http://arxiv.org/abs/1412.6572

[3] https://en.wikipedia.org/wiki/Bisection_method

[4] https://en.wikipedia.org/wiki/Root-finding_algorithms

[5] https://en.wikipedia.org/wiki/ImageNe

[6] https://www.tensorflow.org/tutorials/generative/adversarial_fgsm



