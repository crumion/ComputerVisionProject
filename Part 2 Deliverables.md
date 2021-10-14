# Part 2 Deliverables | 1st Update

## Methods Applied for data pre-processing and feature extraction

After downloading the images, I had to rename files to reflect the images accordingly. Additionally, I set up local directories so I could iterate over them easily. I also fixed file extensions, renaming all extensions to reflect .jpg (some were jpeg).

After some initial data preparations, I started to run some preliminary models. More particularly, I tested my images using MobileNetV2 to observe some baseline predictions. Some images were incorrectly classified, so I removed them from the dataset.

After running some of the images, I ran some adversarial attacks and observed how the model reacted. Consequently, I used Grad-CAM masking to focus the perturbations onto the last convolutional layer. Explainations and details in the following section.

## Justification of Why

### Softmax Scores ≠ Probability
I decdied to start with MobileNetV2 in evaluating its own trustworthiness. For my project, I'm looking for more explainability than merely a "confidence" score. These confidence scores are simply softmax scores, which are often described as probabilities. However, the softmax score is not a probability, so we must be more sophisticated in the ways we evaluate our models.

### Image Alterations
Since softmax scores are unreliable and don't necessarily reflect the model's prediction probabilty, we need to find other ways of evaluating the model. One approach is to alter the image, and observe differences in softmax scores. However, since these softmax scores do not necessarily reflect the model's probability, this approach wouldn't be particularly as useful. Instead, I would like to progressively alter the image until the model makes an incorrect classification. This method would test each model's robustness given a certain class.

### Perturbations
One of the first common adversarial attacks on nueral networks uses Fast Gradient Signed Method (FGSM), first proposed by Goodfellow et al. FGSM uses the gradients of the loss function with respect to the image image. 

<img width="450" alt="Screen Shot 2021-10-14 at 1 55 57 PM" src="https://user-images.githubusercontent.com/30506411/137370941-35ba6fbc-56a5-4b86-b1b6-a5c4ce5a4ba1.png">



### Grad-CAM
Grad-CAMs have become a popular way to evaluate what each model is "looking" at. Grad-CAMs generate heatmaps using the last convolutional layer. These heatmaps can provide practitioners a visual that describes where the model is focusing its attention at the moment before it makes its prediction.

### Grad-CAM Perturbations
I would like to use Grad-CAMs to focus perturbations to where the model is looking at. At this step, we should see the model less tolerant to changes and more easily flipped to incorrect classifications. Next, I would like to flip the Grad-CAM to focus only on the background of the image. We expect to see the model more tolerate to perturbations than the Grad-CAM object perturbations, and the general perturbations. If we repeat these steps with many samples from each class, we should get a smooth distribution of general perturbations, Grad-CAM object-focused perturbations, and Grad-CAM background-focused perturbations. By using these framework, we can apply mathematical operatons to get a score. This score determines the tolerance for each model's classifciation. For example, some classes may have extremely high confidence (> 90%), but with a slight change to the image it will flip to an incorrect classification. Therefore, the confidence score is misleading, but our trustworthy metric wll capture the model's sensitivity to these changes, which helps explain trusthworthiness in the model's decisions.


## A few illustrations demonstrating how your methods processed training data

## Preliminary Results
<img width="600" alt="perturbations_graph" src="https://user-images.githubusercontent.com/30506411/137355770-3873f5a2-1cc1-4f83-a046-dca35aa28c76.png">

## Grad-CAM Example

### First Convolutional Layer
<img width="468" alt="bullfrog1" src="https://user-images.githubusercontent.com/30506411/137356367-55cba57f-8d8c-42b6-8f56-8bf0706effd9.png">

### Hand Picked in Between Samples
<img width="468" alt="bullfrog2" src="https://user-images.githubusercontent.com/30506411/137356404-56d1e07d-e633-4682-8c2a-97d933dff98a.png">

<img width="468" alt="bullfrog3" src="https://user-images.githubusercontent.com/30506411/137356423-0ad0e49d-93c8-4477-8287-166d267b8d39.png">

### Last Convolutional Layer
<img width="468" alt="bullfrog4" src="https://user-images.githubusercontent.com/30506411/137356445-b2f51f14-5a61-4f7b-b243-5e0e324499dc.png">


## Push current codes in conjunction with report

Current codes are pushed in the repo.
Perturbation.py reflects entire image perturbations.


## References
[1]	R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” Int. J. Comput. Vis., vol. 128, no. 2, pp. 336–359, Feb. 2020, doi: 10.1007/s11263-019-01228-7.

[2]	I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and Harnessing Adversarial Examples,” ArXiv14126572 Cs Stat, Mar. 2015, Accessed: Oct. 14, 2021. [Online]. Available: http://arxiv.org/abs/1412.6572



