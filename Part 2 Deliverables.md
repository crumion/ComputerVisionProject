# Part 2 Deliverables | 1st Update

## Overview | Methods Applied for data pre-processing and feature extraction

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
Source: https://www.tensorflow.org/tutorials/generative/adversarial_fgsm

These perturbations can be focused to different parts of the image using other methods like Grad-CAM.

### Grad-CAM Heatmaps, Perturbations
Grad-CAMs have become a popular way to evaluate what each model is "looking" at. Grad-CAMs generate heatmaps using the last convolutional layer. These heatmaps can provide practitioners a visual that describes where the model is focusing its attention at the moment before it makes its prediction [1]. I would like to use Grad-CAMs to focus perturbations to where the model is looking at. At this step, we should see the model less tolerant to changes and more easily flipped to incorrect classifications. Next, I would like to flip the Grad-CAM to focus only on the background of the image. We expect to see the model more tolerate to perturbations than the Grad-CAM object perturbations, and the general perturbations. If we repeat these steps with many samples from each class, we should get a smooth distribution of general perturbations, Grad-CAM object-focused perturbations, and Grad-CAM background-focused perturbations. By using these framework, we can apply mathematical operations to get a trustworthy score. This score determines the tolerance for each trained models's classes. For example, some classes may have extremely high confidence (> 90%), but with a slight change to the image it will flip to an incorrect classification. Therefore, the confidence score is misleading, but our trustworthy metric will capture the model's sensitivity to these changes and give us appropiate model trust.


## A few illustrations demonstrating how your methods processed training data

## Preliminary Results
<img width="600" alt="perturbations_graph" src="https://user-images.githubusercontent.com/30506411/137355770-3873f5a2-1cc1-4f83-a046-dca35aa28c76.png">

This graph shows how the model's confidence scores changes at different epsilon levels. Perhaps the most intriguing part is the model gets more "confident" with its incorrect classifications. Ultimately, this graphic shows that the confidence scores are not a reliable metric. 

## Grad-CAM Example

### First Convolutional Layer
<img width="468" alt="bullfrog1" src="https://user-images.githubusercontent.com/30506411/137356367-55cba57f-8d8c-42b6-8f56-8bf0706effd9.png">

### Hand Picked in Between Layers
<img width="468" alt="bullfrog2" src="https://user-images.githubusercontent.com/30506411/137356404-56d1e07d-e633-4682-8c2a-97d933dff98a.png">

<img width="468" alt="bullfrog3" src="https://user-images.githubusercontent.com/30506411/137356423-0ad0e49d-93c8-4477-8287-166d267b8d39.png">

### Last Convolutional Layer
<img width="468" alt="bullfrog4" src="https://user-images.githubusercontent.com/30506411/137356445-b2f51f14-5a61-4f7b-b243-5e0e324499dc.png">

This heatmap is used to increase perturbation intensity around the object.

## Push current codes in conjunction with report

Current codes are pushed in the repo.
Perturbation.py reflects entire image perturbations. Repo will be updated to relfect additional updates and changes to the project.

## Acknowledgment
This work is an adaptation of an ongoing project through the Computer Vision Research Lab at the University of Notre Dame, advised by Dr. Adam Czajka.

## References
[1]	R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” Int. J. Comput. Vis., vol. 128, no. 2, pp. 336–359, Feb. 2020, doi: 10.1007/s11263-019-01228-7.

[2]	I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and Harnessing Adversarial Examples,” ArXiv14126572 Cs Stat, Mar. 2015, Accessed: Oct. 14, 2021. [Online]. Available: http://arxiv.org/abs/1412.6572





