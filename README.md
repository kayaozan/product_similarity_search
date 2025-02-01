# Product Similarity Search
This aim of this project is to build a pipeline to find similar products to a given image. Let us begin by breaking down the challenges.

## Exploration
### Dataset
The dataset of the products consists of shoe images. Most of these images belong to a pair of shoes on a white background like the one below.

![image](https://github.com/user-attachments/assets/db6f7876-ff9c-4b04-a451-4b7d349bbee4)

Before even comparing these to other images, they need to be processed, so that the images only contain shoes and nothing else.

First solution that I thought of was to crop the non-white section. This works very nicely, but remember that I said "most" of these images are like the one above.

Some of the images don't have any pattern. They are shot in various environments while shoes being worn by a person, like below.

![image](https://github.com/user-attachments/assets/bc3b3ee0-3a8d-4334-b8e5-b38b7340155d)

There was no way to make "crop the non-white" work for this kind of images. So that plan was off the table, I needed a way to process every one of the images.

Then I thought, why not use an object detection model to process them? I figured I was also going to need it to process images the user will send. So the search for an object detector began.

### Object Detection
I've examined a lot of object detection models. Some of them was able to detect shoes in images if they are worn by a person, but all of them failed to detect them mostly in close-up shots, so are most of the dataset.

I was about to give up and train my own model, then I found YOLO, specifically the [YOLOv8](https://docs.ultralytics.com/datasets/detect/open-images-v7/) trained on [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html).

I'd tried YOLO before, but this one in particular was able to detect every single one of the shoes in the images.

### Feature Extraction
Once the shoes are detected in images, they are cropped and ready to be preprossed before similarity check. At this point, a feature extraction model is needed.

I considered using YOLO for extracting features. That would also mean running one model instead of two, lowering the resources used, but I experienced that the features extracted by YOLO was unsatisfactory.

So I've gone back to one of the trusted models, [ResNet](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html). The extracted features was rich enough to find similarities in images.

## Pipeline
Now that the parts of the pipeline are ready, let us list the steps of the process. To build the dataset of features:

- Go through all images in the dataset
- Run the object detection on images
- Return the desired objects only (for this project, shoes)
- Crop the objects if detected
- Extract the features of the cropped image
- Save the features

After building the dataset of features, it is ready to be compared to any given image. To search for similarity:

- Ask the user for an image of shoes
- Run the object detection and crop the shoes
- Extract the features
- Compute the similarity of the features and the dataset of features
- Return the products sorted by highest similarity score

## [Streamlit](https://streamlit.io/) App
To be able to send images as user and check the results, a platform was necessary. Streamlit is a library to run Python codes in a web browser without any knowledge about web design.

I've included a script named `object_detection_app.py` in the repository. It consists of a simple web page design that lets user to send an image URL.

Similarity comparison is run in the background and the results are shown. Here is a screenshot of the page. Notice the detected shoe in the image on the left.

![image](https://github.com/user-attachments/assets/0701f647-2403-4e23-8eae-a3985416260f)

Lastly, I've included a Dockerfile to run everything in a container. Exposing a port in the file lets streamlit to be run in the local web browser.
