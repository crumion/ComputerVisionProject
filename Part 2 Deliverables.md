# Part 2 Deliverables | 1st Update

1. Methods Applied for data pre-processing and feature extraction
After downloading the images, I had to rename files to reflect the images accordingly. Additionally, I set up local directories so I could iterate over them. I also fixed file extensions, renaming all extensions to reflect .jpg (some were jpeg).

2. A short justification why you decided to use these algorithms
I decdied to start with MobileNetv2 in evaluating its own trustworthiness. For my project, I'm looking for more explainability than merely a "confidence" score. These confidence scores are simply softmax scores, which are often described as probabilities. However, the softmax score is not a probability, so we must be more sophisticated in the ways we evaluate our models.

Since softmax scores are unreliable and don't necessarily reflect the model's prediction probabilty, we need to find other ways of evaluating the model. One approach is to alter the image, and observe differences in softmax scores. However, since these softmax scores do not necessarily reflect the model's probability, this approach wouldn't be particularly as useful. Instead, I would like to progressively alter the image until the model makes an incorrect classification. This method would test each model's robustness given a certain class.

Grad-CAMs have become a popular way to evaluate what each model is "looking" at. Grad-CAMs generate heatmaps using the last convolutional layer. These heatmaps can provide practitioners a visual that describes where the model is focusing its attention at the moment before it makes its prediction.

I would like to use Grad-CAMs to focus perturbations to where the model is looking at. At this step, we should see the model less tolerant to changes and more easily flipped to incorrect classifications. Next, I would like to flip the Grad-CAM to focus only on the background of the image. We expect to see the model more tolerate to perturbations than the Grad-CAM object perturbations, and the general perturbations. If we repeat these steps with many samples from each class, we should get a smooth distribution of general perturbations, Grad-CAM object-focused perturbations, and Grad-CAM background-focused perturbations. By using these framework, we can apply mathematical operatons to get a score. This score determines the tolerance for each model's classifciation. For example, some classes may have extremely high confidence (> 90%), but with a slight change to the image it will flip to an incorrect classification. Therefore, the confidence score is misleading, but our trustworthy metric wll capture the model's sensitivity to these changes, which helps explain trusthworthiness in the model's decisions.


3. A few illustrations demonstrating how your methods processed training data, for instance segmentation results
<img width="600" alt="perturbations_graph" src="https://user-images.githubusercontent.com/30506411/137355770-3873f5a2-1cc1-4f83-a046-dca35aa28c76.png">

<img width="468" alt="bullfrog1" src="https://user-images.githubusercontent.com/30506411/137356367-55cba57f-8d8c-42b6-8f56-8bf0706effd9.png">

<img width="468" alt="bullfrog2" src="https://user-images.githubusercontent.com/30506411/137356404-56d1e07d-e633-4682-8c2a-97d933dff98a.png">

<img width="468" alt="bullfrog3" src="https://user-images.githubusercontent.com/30506411/137356423-0ad0e49d-93c8-4477-8287-166d267b8d39.png">

<img width="468" alt="bullfrog4" src="https://user-images.githubusercontent.com/30506411/137356445-b2f51f14-5a61-4f7b-b243-5e0e324499dc.png">


4. Push current codes in conjunction with report


