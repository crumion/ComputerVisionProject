# Part 4 deliverables (Final report)


The aim of this last part of the semester project is to test your solutions on unknown data. By "unknown data" I mean a sequestered set of samples, not used (or seen) when you were designing your method in parts 2 and 3 of the project. Here is a general list of deliverables for Part 4:

A report (no page limit, but try to be concise; 3-4 pages should suffice) or a separate (from Parts 1-3) readme GitHub doc that includes:

## Report

- Description of the test database you collected or downloaded: What is the size of the database? What is different when compared to the training and validation subsets? Why you believe these differences are sufficient to test your final programs? (2 points)

We used the validation set of ImageNet to evaluate the adversarial robustness of a model (Note: This dataset blurs the faces of people that are featured in object detection, which has an affect to some degree on the classification of the network). This dataset recorded a much higher overall accuracy than Tiny-ImageNet (68.27%, 24.11%), which was used previously in Part 3. We expected this performance difference for two reasons. The first reason is that our model, MobileNetV2, was pretrianed on ImageNet, not Tiny-ImageNet. Second, ImageNet had larger file sizes and better quality images. We actually had to downscale these images before feeding it through the model at the standard (224, 224) dimensions.



- A classification accuracy achieved on the test set. Use the same metrics as in part 3. (3 points)

Due to time and computational constraints, we randomly sampled 5 out of the 50 images per class with replacement. Out of 5,000 images, 124 images threw ValueErrors and had to be skipped. While the error is somewhat ambigous, we suggest a preprocessing step failed to format the image correctly (either incorrect sizing, incorrect channels, etc.). These errors were labeled and recorded in the log files. After throwing out those images, we were left with 4876 images for our test set. We also removed images that the model could not correctly classify with zero adversarial attack. The model correctly predicted the ground truth classification 68.27% of the time (3329/4876). The remaining 3329 images were perturbed until an incorrect classification was obtained.

In part 3, we measured robustness by the average first failure of the model (measured by epsilon, or step values). We only used the FSGM, which perturbs the entire image at equal rates. For out final submission, we used Grad-Cams to focus those perturbations so that the images are perturbed unequally onto the object or the background. For example, we expected to see higher average epsilon failures for the background, and the contrary for object perturbations. The regular FGSM serves as a rough middle ground for a comparison of the two types of attacks.




- Most of you should see worse results on the test set when compared to the results obtained on train/validation sets. You should not be worried about that, but please provide the reasons why your solution performs worse (with a few illustrations, such as pictures or videos, what went wrong). What improvements would you make to lower the observed error rates? (5 points)



## Presentation

Imagine you want to present your final program to a friend (or investor). And this presentation should be short and illustrative (that is: show what you did and how good it is). Prepare a short video, or a short slideshow with pictures, presenting how your final program works. Here are good examples prepared by former CV students. (5 points)


## Final Code Base

Push your final (to be graded) codes realizing what you mention in the report to GitHub, along with instructions how to run them (either Adam or Siamul will do it to see how the final solution works on test data). Your program(s) should pick one example from the test set (please attach this sample to your codes) and present the processing result. We should be able to run your programs without any edits -- please double check before submission that the codes are complete. (5 points)
