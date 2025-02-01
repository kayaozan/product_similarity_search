WORK IN PROGRESS

# Product Similarity Search
This aim of this project is to build a pipeline to find similar products to a given image. Let us break down the challenges of this project.

## Exploration
### Dataset
The dataset of the products consists of shoe images. Most of these images belong to a pair of shoes on a white background like the one below.

![image](https://github.com/user-attachments/assets/db6f7876-ff9c-4b04-a451-4b7d349bbee4)

Before even comparing these to other images, they need to be processed, so that the images only contain shoes and nothing else.

First solution that I thought of was to crop the non-white section. This works very nicely, but remember that I said "most" of these images are like the one above.

Some of the images don't have a pattern. They are shot in various environments while shoes being worn by a person, like below.

![image](https://github.com/user-attachments/assets/bc3b3ee0-3a8d-4334-b8e5-b38b7340155d)

There was no way to make "crop the non-white" work for this kind of images. So that plan was off the table, I needed a way to process every one of the images.

Then I thought, why not use an object detection model to process them? I figured I was also going to need it to process images the user will send. So the search for an object detector began.

### Object Detection
I've examined a lot of object detection models. Some of them was able to detect shoes in images if they are worn by a person, but all of them failed to detect them mostly in close-up images, so are most of the dataset.

I was about to give up and train my own model, then I found YOLO, specifically the [YOLOv8](https://docs.ultralytics.com/datasets/detect/open-images-v7/) trained on [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html).

I'd tried YOLO before, but this one in particular was able to detect every single one of the shoes in the images.

## Pipeline
Key components are 
