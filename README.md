WORK IN PROGRESS

# Product Similarity Search
This aim of this project is to build a pipeline to find similar products to a given image. Let us break down the challenges of this project.

## Challenges
The dataset of the products consists of shoe images. Most of these images belong to a pair of shoes on a white background like the one below.

![image](https://github.com/user-attachments/assets/db6f7876-ff9c-4b04-a451-4b7d349bbee4)

Before even comparing these to other images, they need to be processed, so that the images only contain shoes and nothing else.

First solution that I thought of was to crop the white section, the background. This works very nicely, but remember that I said "most" of these images are like the one above.

Some of the images doesn't have a pattern. They are shot in different environments while shoes being worn by a person, like below.

![image](https://github.com/user-attachments/assets/b50467ba-cf29-4695-952c-a7384cc79174)

