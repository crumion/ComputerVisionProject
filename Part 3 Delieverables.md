# Computer Vision 1 Semester Project


## Dataset

I will be using ImageNet for my dataset. ImageNet contains over 14 million annotated images across over 20,000 categories. Many proposed models for my project are trained on ImageNet. ImageNet has built arguments to split the dataset into training, validation, and testing sets, though these will likely not be as relevant to my project.
(source: https://en.wikipedia.org/wiki/ImageNet)

## Proposed Solution

I will be modifying some of the work I've been doing on the Crane research project, specifically within the context of trustworthiness in AI. Trustworthy AI involves frameworks, metrics, or tools that help us trust the conclusions of deep learning models. These techniques help break down "black boxes" and make models more transparent, more graceful to failures, and robust. In a world where models are being used in high stakes situations and lives may be at stake, it's essential that we build trustworthy and explainable features into AI so that we can effectively work in unison on more expansive projects in the future.

For my semester project, I will be forming a reliability metric to access the models based on random, out of class inputs. My previous work has involved adding random perturbations onto different parts of an image, and observing the differences in model confidence. Using object masking, I have added perturbations to the background, object, and entire image to compare confidence scores and measure different ways the model classify an image. I have used VGG16, VGG19, and Xception and evaluated their robustness to these adversarial attacks. For my semester project, I plan to use these models in more diverse ways, eventually creating a set of metrics to evaluate their performance and the various ways we can access their trustworthiness. In theory, by checking their responses to different situations (perturbations, rotations, etc.) we can formulate a trust score, and thus we can have a better measure in their ability to perform tasks accurately than merely observing confidence scores.