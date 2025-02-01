import torch
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.detect import bbox_iou
from utils.image import image_by_URL, get_non_white_bounding_box, pad_to_square


# Load the configurations.
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
config = config['DEFAULTS']

# Read the images of dataset.
images = pd.read_pickle('images_dataset')

# Initialize the lists for features and log messages.
features = []
logs = []

# Load the detection model.
detection_model = YOLO(config['DetectionModel'])

# Define the objects to be detected.
target_names = config['TargetNames'].split(',')
target_IDs = [key for key, value in detection_model.names.items()
             if value in target_names]

# Load the preprocessor.
weights = ResNet50_Weights.DEFAULT
preprocessor = weights.transforms()

# Load the feature extractor model.
feature_extractor = resnet50(weights=weights)
# Replace the final layer with an identity layer,
# so that it will act as a feature extractor.
feature_extractor.fc = torch.nn.Identity()
feature_extractor.eval()

# Define the max and min values of Intersection of Union.
maxIoU = 0.9
minIoU = 0.1

# Loop through the images of dataset.
for rowNum in tqdm(range(len(images))):

    # Skip if the url cannot be opened.
    try:
        img = image_by_URL(images.iloc[rowNum]['URL'])
    except:
        logs.append('URLError')
        features.append(np.nan)
        continue

    # Run inference on the image.
    results = detection_model.predict(source=img,
                                      verbose=False,
                                      agnostic_nms=True,
                                      conf=0.25,
                                      iou=0.1,
                                      classes=target_IDs)[0]

    # Get the bounding boxes and confidence scores
    bboxes = results.boxes.xyxy.numpy()
    confs = results.boxes.conf.numpy()

    if len(bboxes) > 0:
        # Select the bounding box of highest confidence.
        img_pick = img.crop(bboxes[np.argmax(confs)])
        logs.append('Detected')
    else:
        # If there is no detection, crop the white background.
        bbox_NonWhite = get_non_white_bounding_box(img)

        if bbox_NonWhite is None:
            # Skip if the image is all white.
            features.append(np.nan)
            logs.append('WhiteError')
            continue
        else:
            # Check for the IoU of the cropped area and the image.
            IoU_NWImage = bbox_iou(bbox_NonWhite,
                                   [0,0,img.size[0],img.size[1]])

            if IoU_NWImage < maxIoU and IoU_NWImage > minIoU:
                img_pick = img.crop(bbox_NonWhite)
                logs.append('Cropped')
            else:
                # The cropped area is large or small compared to the image.
                # large: there is no significant cropping occurred.
                # small: the object shouldn't be that small.
                features.append(np.nan)
                logs.append('CropError')
                continue
    
    # Complete the image to square and send to preprocessor.
    img_padded = pad_to_square(img_pick)
    img_prep = preprocessor(img_padded).unsqueeze(0)

    # Extract features and save.
    with torch.no_grad():
        featuresImage = feature_extractor(img_prep).squeeze(0).numpy()

    features.append(featuresImage)

# Save the features and logs.
pd.to_pickle(features,'features_dataset')
pd.to_pickle(logs,'logs_dataset')
