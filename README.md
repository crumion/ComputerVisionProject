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

One of the first common adversarial attacks on nueral networks uses Fast Gradient Signed Method (FGSM), first proposed by Goodfellow et al. FGSM uses the gradients of the loss function with respect to the image [2]. These perturbations are one way to fool the model, and observe how quickly it breaks by adjusting the epsilon. If the model is not fooled until high levels of epsilon, then it is trustworthy. If the model is fooled rather quickly, we cannot be confident in its deployment on test data. Below are some details regarding the perturbation formula:

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

The model correctly classified 24.11% of the validation images. Some of the images were incorrectly formatted which threw a value error, so I added exception handling to circumnavigate this error. 168 images were not tested and thrown out because of this error. Of the 10,000 images, 2411 remained correctly formatted, classified, and ready for adversarial attacks.

I used the Fast Gradient Sign Method to generate adversarial attacks on each image. I started at 0 epsilon and used step sizes of 0.00001 until the model predicted an incorrect classification. Since this was a linear search, my script took days to run only a fraction of the data set (I have a CRC long script running it as of writing. My first script broke over an input ValueError, but I have committed the results I got up until that point). Instead of taking a smaller sample size, I decided to use a faster search algorithm. I wrote a bisectional method algorithm to find the smallest epsilon, which resulted in much faster speeds [3]. The method involved finding the midpoint of an upper and lower bound, and then testing which side of the bounds a failure occurs. Depending on the success or failure of the prediction, the lower or upper bound would become the middle bound, and a new midpoint would be computed. The algorithm would repeat until the difference between the upper and lower bounds was less than 0.00001 (step value). I was able to run the model over the entire validation set in only 6 hours. The epsilon values differed drastically, ranging from 0.000019 to 0.95. This was to be expected, because each image and classification is completely different. I averaged the epsilon values for each class, so I could access class robustness.

The most "sensitive" class was grasshopper, followed by fly and pizza (0.0000214440918, 0.000431060791, 0.0008377893066). The most "robust" classes were poncho, sock, and confectionery (0.46, 0.37, 0.31). These values tell us the average epsilon value for each class to fail (predict an incorrect classification). These show how far we have to travel in the direction of the gradient (signed method) to maximize the loss and fail the model. If we can easily fool a model, we should not readily trust it's ability to predict on new test sets.


## Room for Improvement

I currently have a pipeline to test how robust each model and its classifiers are for Fast Gradient Sign Method. By generating these distributions, I can compare one model's robustness over another and gain confidence when showing these systems new images. These normal perturbations will be compared with the Grad-Cam Object perturbations and the Grad-Cam Background perturbations. We expect to see a range of average epsilon values across all three tests, which will add into our trusthworthiness metric. For example, if the Background perturbations are close to the Object perturbations, we can conclude the model isn't looking particularly at the object. However, if the average epsilon value is much higher for Background perturbations than for Object perturbations, we can conclude the model and class is more robust. There will be a failure point at each and every model, but the focus is to provide a range of confidence we can place on the model's ability to predict.

Initial trials of the Object and Background perturbations were inconclusive. Charts below show the inconsistencies between robustness, as we observed the opposite affect. Object perturbations were less sensitive to epsilon values, followed by Background and regular perturbations. We are still analyzing and interpreting these findings. These tests were run on a very small data set (n=10), and on a very sensitive class (bullfrog, which is in the top 1% of sensitive classes). Regardless of these findings, the relative pipelines are built and will be performed for the final sections of this project. However, we will be modifying our work going forward and using another way to interpret trustworthiness: heatmaps.

Similarly to Grad-Cams, the gradient can also be visualized as a heatmap. This could be used as a replacement of Grad-Cams to estimate what the network is "looking" at. We wish to find a resemblance to how humans would classify objects. For example, we expect heatmaps to be concentrated around the face to classify a human. For future work, we will attempt to measure the compactness of a heatmap. The heatmaps of a bullfrog above show the differences in concentration the network places on an object (though these are Grad-Cams taken at different convolutional stages). We feel confident that a dense heatmap has higher robustness than one that is spread out, since there are likely only a few close features essential to classification. These concentrations can also be compared to Grad-Cams to see if the last convolutional layer agrees with the gradient (should be the same).

Another component of the pipeline that must be fixed before the final results is the code. The committed code is functional but not optimal. First, functions and classes need to be made and will improve code readability. Second, the code output to text files should be formatted better. I had to spend time data wrangling which could be prevented by better txt outputs. Third, the way the bisectional algorithm is written there is no check to see if the last prediction is equal to the first prediction. In some cases, the final epsilon value will not fully break the model. In these cases, the step value (0.00001) must be added to obtain the true breaking epsilon value. A simple logic statement after the loop will resolve this issue. I would also like to try other root finding methods to see if I can speed up the process. Upon review, the secant method looks promising to try, althrough there are a handful of potential methods [4]. Finally, the code does not stop adversarial attacks for incorrectly classified images. This will greatly improve the speed of the model and test only the necessary images. As always, the code could be threaded or optimized for GPU use. This will allow faster results and more scalable testing of the trustworthy pipeline.

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

# Part 4 deliverables (Final report)

> The aim of this last part of the semester project is to test your solutions on unknown data. By "unknown data" I mean a sequestered set of samples, not used (or seen) when you were designing your method in parts 2 and 3 of the project. Here is a general list of deliverables for Part 4: A report (no page limit, but try to be concise; 3-4 pages should suffice) or a separate (from Parts 1-3) readme GitHub doc that includes: report, presentation, and final codes pushed to GitHub.

# Report

## Description of the Test Database

> Description of the test database you collected or downloaded: What is the size of the database? What is different when compared to the training and validation subsets? Why you believe these differences are sufficient to test your final programs? (2 points)

We used the validation set of ImageNet to evaluate the adversarial robustness of a pretrained model (Note: This dataset blurs the faces of people that are featured in object detection, which has an affect to some degree on the classification of the network). The database has over 50,000 images, 50 images per class. This dataset recorded a much higher overall accuracy than Tiny-ImageNet (68.27%, 24.11%), which was used previously in Part 3. We expected this performance difference for two reasons. The first reason is that our model, MobileNetV2, was pretrained on ImageNet, not Tiny-ImageNet. Second, ImageNet had larger file sizes and better quality images. We actually had to downscale these images before feeding it through the model at the standard (224, 224) dimensions.

Due to time and computational constraints, we randomly sampled 5 out of the 50 images per class with replacement. Out of 5,000 images, 124 images threw ValueErrors and had to be skipped. While the error is somewhat ambiguous, we suggest a preprocessing step failed to format the image correctly (either incorrect sizing, incorrect channels, etc.). These errors were labeled and recorded in the log files. After throwing out those images, we were left with 4876 images for our test set. We also removed images that the model could not correctly classify with zero adversarial attack. The model correctly predicted the ground truth classification 68.27% of the time (3329/4876). The remaining 3329 images were perturbed until an incorrect classification was obtained.


## Accuracy Performance
 > A classification accuracy achieved on the test set. Use the same metrics as in part 3. (3 points)

In part 3, we measured robustness by the average first failure of the model (measured by epsilon, or step values). We only used the Fast Sign Gradient Method (FSGM), which perturbs the entire image at equal rates. For out final submission, we used Grad-Cams to focus those perturbations so that the images are perturbed unequally onto the object or the background for each epsilon step. We expected to see higher average epsilon failures for the background, and the opposite for object perturbations since the model should be looking at the object its classifying.

The regular FGSM serves as a rough middle ground for a comparison of the two types of attacks. The most robust classes for regular FGSM attacks were wool (0.68), doormat (0.53), and peacock (0.51). The most sensitive classes were petri dish (9.20E-06), cassette player (0.0002), langur (0.0003), and dough (0.0005).

For Background focused FGSM perturbations, the most robust classes were doormat (0.92), barrow (0.73), maze (0.71), and then peacock (0.67). The most sensitive background FGSM classes were petri dish (8.88E-06), cassette player (0.00019), and dough (0.00049).

For Object focused FGSM perturbations, the most robust classes were patio (0.80), jigsaw puzzle (0.75), bonnet (0.72), and wool (0.69). The most sensitive object focused classes were petri dish (1.08E-05), langur (0.00012), cassette player (0.00021), and dough (0.00066).

If we compare these results to regular FGSM epsilon values from Tiny-ImageNet, we can see that our test database was much more robust (The top 3 most robust classes failed at 0.68, 0.53, 0.51 compared to 0.46, 0.37, 0.31). The test set also recorded much higher levels of sensitivity (The top 3 most sensitive classes failed at 9.20E-06, 0.0002, 0.0003 compared to 0.00002, 0.0004, and 0.0008), meaning even our most sensitive classes were more robust. Overall, these results are indicative of a better trained and more robust model. Again, these results were expected due to the training of ImageNet and the better quality images.


## Analysis of Results, Further Improvements

> Most of you should see worse results on the test set when compared to the results obtained on train/validation sets. You should not be worried about that, but please provide the reasons why your solution performs worse (with a few illustrations, such as pictures or videos, what went wrong). What improvements would you make to lower the observed error rates? (5 points)

Since we our evaluating the trustworthiness of a pretrained model, it will be difficult to compare our results exactly with part 3. These results are expected to be more expansive and better representative than our previous developments. However, these results shed light on how the model is making its classifications and the model's performance as a whole. First, our results show a distribution of average failed epsilon values. These can be understood as how "robust" the models are to inputs that maximize their loss functions w.r.t. each class label. Using Gradient Class Activation Mappings, we focused the perturbations unequally to reflect greater epsilon values (or step values) for either the object or the background. As stated previously, we expected the background distributions to be shifted towards the right, and the object distributions to be shifted towards the left (in comparison to the regular FGSM distributions). This is exactly what we saw in our Part 4 results. We can use these distributions to develop a metric to assess the changes in epsilon values from the object to the background. For a robust class, we hope to see a great difference between these two values. Once we have a distribution of the classification space, we can add our metrics to assess the robustness of each and every class.

These test scripts should be run 8-12 times to get a proper distribution of the relative robustness. Additionally, these distributions can be compared to other popular models to access the trustworthiness of each model and its training. We acknowledge that our experiments are one angle of addressing trustworthiness, and thus incomplete alone. However, we believe this is a starting point to evaluate model trustworthiness and ensure intelligently trained models for the future. 


### Most Robust Regular FGSM Classes
![Most Robust Regular FGSM Classes](https://user-images.githubusercontent.com/30506411/145866464-0a23f8f3-a0e4-4028-b448-72ac691b0c1f.png)

### Most Sensitive Regular FGSM Classes
![Most Sensitive Regular FGSM Classes](https://user-images.githubusercontent.com/30506411/145866470-ced6b53a-94bd-4a58-821c-6d1548e39e97.png)

### Most Robust Background FGSM Classes
![Most Robust Background FGSM Classes](https://user-images.githubusercontent.com/30506411/145866387-a4ef1349-b87a-405d-a1d9-db0521ea1af3.png)

### Most Sensitive Background FGSM Classes
![Most Sensitive Background FGSM Classes](https://user-images.githubusercontent.com/30506411/145866403-6a556db6-7242-45ca-b995-fc32551088d0.png)

### Most Robust Object FGSM Classes
![Most Robust Object FGSM Classes](https://user-images.githubusercontent.com/30506411/145866357-b92d4abe-27ea-427e-8e7d-44ef86b172b6.png)

### Most Sensitive Object FGSM Classes
![Most Sensitive Object FGSM Classes](https://user-images.githubusercontent.com/30506411/145866372-6edae8aa-d0a6-4810-b3c7-54243aabe93f.png)

### Grad-Cam Focusing Perturbations, all classes
![Grad-Cam Focusing Perturbations (2)](https://user-images.githubusercontent.com/30506411/145866608-32e40f10-4280-47ff-8f6e-a1c315bde3ce.png)

### Grad-Cam Focusing Perturbations, all classes and adjusted x axis
![Grad-Cam Focusing Perturbations (1)](https://user-images.githubusercontent.com/30506411/145866624-f345ce58-aee9-41aa-b3fb-2954c7d39eb8.png)

### Difference between Background and Object Perturbations
![difference](https://user-images.githubusercontent.com/30506411/145876287-5514ca20-9d94-4eab-a671-3ca7d97b2205.png)

### Object Dependent Classes
![object_dependent](https://user-images.githubusercontent.com/30506411/145876657-eee60d93-f63e-463f-923f-918a7c603eaf.png)

### Background Dependent Classes
![background_dependent](https://user-images.githubusercontent.com/30506411/145876676-3081527d-6976-49b2-b506-b3e46bb2c8a3.png)


# Presentation

> Imagine you want to present your final program to a friend (or investor). And this presentation should be short and illustrative (that is: show what you did and how good it is). Prepare a short video, or a short slideshow with pictures, presenting how your final program works. Here are good examples prepared by former CV students. (5 points)

Presentation can be found at:
https://www.youtube.com/watch?v=9qEd87Uzbs0

Presented slides can be found at:
https://www.canva.com/design/DAEydQlWS3M/NRyE-vaFO27QraUnP2JbmQ/view?utm_content=DAEydQlWS3M&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton

A more detailed presentation can be found at:
https://docs.google.com/presentation/d/1Ip74d2KpsDbfBxkAkUihtWXDJZ-Sb4dFeYxycmh1Tlc/edit?usp=sharing

A slide deck reflecting the weekly progress updates can be found at:
https://docs.google.com/presentation/d/107tTgQ9NW03ETU0Qzd20T9sshafNEGhtClaim0LZ79w/edit?usp=sharing


# Final Code Base

> Push your final (to be graded) codes realizing what you mention in the report to GitHub, along with instructions how to run them (either Adam or Siamul will do it to see how the final solution works on test data). Your program(s) should pick one example from the test set (please attach this sample to your codes) and present the processing result. We should be able to run your programs without any edits -- please double check before submission that the codes are complete. (5 points)

## Folder Structure Diagram
![Screenshot from 2021-12-13 15-20-48](https://user-images.githubusercontent.com/30506411/145882994-49bfc24e-4bbc-44f1-9ed8-a4ff47936263.png)

The example codes can be found under ./Example Codes in the main directory. The test scripts can be found under ./test scripts. The above graphic represents the appropriate folder structure and files to run the example code. Due note there are a few changes. The first change in these codes are the second for loop, where the line:

for file in resample(files, n_samples=5, replace=False, random_state=1):

has been commented out and changed to:

for files in files:

This line reflects only one sample class and one sample image, instead of sampling 5 images from every class. Additionally, the val_blurred directory has only a sample class and image, not the entire ImageNet validation set. The only difference between the sample codes and test codes are the above line and expanded database.

The conda environment used to run these scripts can be found under adv_requirements.txt. This shows all necessary dependencies and can ease the burden of creating the appropriate conda environments.

## File Details
- main.py files are the appropriate test scripts. Each main file has be organized under its respective attack (background, object, regular).
- imagenet_labels.txt reflect the appropriate classes, numeric index, name index, and respective image labels
- adx_requirements.txt shows all the necessary conda dependencies
- n15075141 shows the directory for toilet_tissue
- ILSVRC2012_val_00006482.jpg shows a sample image for toilet_tissue

## Extended Notes on the Code Base
- Each directory has a main.py file and a labels database. The main.py files are the same except for changes in the use of grad-cam (none, object focused, background focused), and the change in adversarial images (adv_x = image + eps * perturbations OR adv_x = image + eps * perturbations * cam_weights).
- All scripts employ the root finding method BrentH, taken from scipy. This is a variation of Brent’s method that uses hyperbolic extrapolation instead of inverse quadratic extrapolation. Brent’s method uses a combination of bisection, secant, and inverse quadratic interpolation.
- For focused object and background perturbations, the following adjustments were made. First, grad-cams were focused around darker parts of the region, so lower RGB values. Grad-Cams were fine for background focused (cam_weights), but needed to be rescaled for object focused (cam_weights = 255 - cam_weigts). Additionally, since we modified the adv_x line, cam_weights had to be scaled to a mean of 1 (adv_x = image + eps * perturbations * cam_weights_normalized). This step was necessary since the previous adv_x multiplication step didn't feature any extra weights. This scaling steps allowed us to be mathematically fair in comparing cam generated weights (object and background) to regular FGSM attacks.

## Acknowledgment
This work is an adaptation of an ongoing research project through the Computer Vision Research Lab at the University of Notre Dame, advised by Dr. Adam Czajka.

## References
[1]	R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” Int. J. Comput. Vis., vol. 128, no. 2, pp. 336–359, Feb. 2020, doi: 10.1007/s11263-019-01228-7.

[2]	I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and Harnessing Adversarial Examples,” ArXiv14126572 Cs Stat, Mar. 2015, Accessed: Oct. 14, 2021. [Online]. Available: http://arxiv.org/abs/1412.6572

[3] https://en.wikipedia.org/wiki/Bisection_method

[4] https://en.wikipedia.org/wiki/Root-finding_algorithms

[5] https://en.wikipedia.org/wiki/ImageNe

[6] https://www.tensorflow.org/tutorials/generative/adversarial_fgsm



