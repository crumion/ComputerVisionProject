# Part 4 deliverables (Final report)


> The aim of this last part of the semester project is to test your solutions on unknown data. By "unknown data" I mean a sequestered set of samples, not used (or seen) when you were designing your method in parts 2 and 3 of the project. Here is a general list of deliverables for Part 4: A report (no page limit, but try to be concise; 3-4 pages should suffice) or a separate (from Parts 1-3) readme GitHub doc that includes: report, presentation, and final codes pushed to GitHub.

## Report

### Description of the Test Database

> Description of the test database you collected or downloaded: What is the size of the database? What is different when compared to the training and validation subsets? Why you believe these differences are sufficient to test your final programs? (2 points)

We used the validation set of ImageNet to evaluate the adversarial robustness of a model (Note: This dataset blurs the faces of people that are featured in object detection, which has an affect to some degree on the classification of the network). This dataset recorded a much higher overall accuracy than Tiny-ImageNet (68.27%, 24.11%), which was used previously in Part 3. We expected this performance difference for two reasons. The first reason is that our model, MobileNetV2, was pretrianed on ImageNet, not Tiny-ImageNet. Second, ImageNet had larger file sizes and better quality images. We actually had to downscale these images before feeding it through the model at the standard (224, 224) dimensions.

Due to time and computational constraints, we randomly sampled 5 out of the 50 images per class with replacement. Out of 5,000 images, 124 images threw ValueErrors and had to be skipped. While the error is somewhat ambigous, we suggest a preprocessing step failed to format the image correctly (either incorrect sizing, incorrect channels, etc.). These errors were labeled and recorded in the log files. After throwing out those images, we were left with 4876 images for our test set. We also removed images that the model could not correctly classify with zero adversarial attack. The model correctly predicted the ground truth classification 68.27% of the time (3329/4876). The remaining 3329 images were perturbed until an incorrect classification was obtained.


### Accuracy
 > A classification accuracy achieved on the test set. Use the same metrics as in part 3. (3 points)

In part 3, we measured robustness by the average first failure of the model (measured by epsilon, or step values). We only used the FSGM, which perturbs the entire image at equal rates. For out final submission, we used Grad-Cams to focus those perturbations so that the images are perturbed unequally onto the object or the background. For example, we expected to see higher average epsilon failures for the background, and the contrary for object perturbations.

The regular FGSM serves as a rough middle ground for a comparison of the two types of attacks. The most robust classes for regular FGSM attacks were wool (0.68), doormat (0.53), and peacock (0.51). The most sensitive classes were petri dish (9.20E-06), cassette player (0.0002), langur (0.0003), and dough (0.0005).

For Background focused FGSM perturbations, the most robust classes were doormat (0.92), barrow (0.73), maze (0.71), and then peacock (0.67). The most sensitive background FGSM classes were petri dish (8.88E-06), cassette player (0.00019), and dough (0.00049).

For Object focused FGSM perturbations, the most robust classes were patio (0.80), jigsaw puzzle (0.75), bonnet (0.72), and wool (0.69). The most sensitive object foucsed classes were petri dish (1.08E-05), langur (0.00012), cassette player (0.00021), and dough (0.00066).


### Analysis of Results, Further Improvements

> Most of you should see worse results on the test set when compared to the results obtained on train/validation sets. You should not be worried about that, but please provide the reasons why your solution performs worse (with a few illustrations, such as pictures or videos, what went wrong). What improvements would you make to lower the observed error rates? (5 points)

Since we our evaluating the trusthworthiness of a pretrained model, it will be difficult to compare our results with part 3. These results are expected to be more expansive and better representative than our previous developments. However, these results shed light on how the model is making its classifications and what that says about the model as a whole. First, our results show a distribution of average failed epsilon values. These can be understood as how "robust" the models are to inputs that maximize their loss functions w.r.t. each class label. Using Gradient Class Activiation Mappings, we focused the perturbations unequally to reflect greater epsilon values (or step values) for either the object or the background. As stated previously, we expected the background distributions to be shifted towards the right, and the object distributions to be shifted towards the left (in comparison to the regular FGSM distributions). This is exactly what we saw in our Part 4 results. We can use these distributions to develop a metric to assess the changes in epsilon values from the object to the background. For a robust class, we hope to see a great difference between these two values. 


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


## Presentation

> Imagine you want to present your final program to a friend (or investor). And this presentation should be short and illustrative (that is: show what you did and how good it is). Prepare a short video, or a short slideshow with pictures, presenting how your final program works. Here are good examples prepared by former CV students. (5 points)


## Final Code Base

> Push your final (to be graded) codes realizing what you mention in the report to GitHub, along with instructions how to run them (either Adam or Siamul will do it to see how the final solution works on test data). Your program(s) should pick one example from the test set (please attach this sample to your codes) and present the processing result. We should be able to run your programs without any edits -- please double check before submission that the codes are complete. (5 points)
