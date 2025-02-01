import streamlit as st
from streamlit_extras.image_selector import image_selector

import torch
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO

import pandas as pd

# User defined packages
from utils.image import image_by_URL, pad_to_square
from utils.detect import find_closest
from utils.webapp import print_all_colors


# Load the configurations.
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

# Default page configs.
st.set_page_config(layout='wide',
                   initial_sidebar_state='collapsed')

# The key point about streamlit is that the code runs every time a change is made on the page.
# By storing the objects in session_state, streamlit doesn't need to reload them.
# That is why the following initializations are done.

# Read data.
if 'images_dataset' not in st.session_state:
    st.session_state.images_dataset = pd.read_pickle('images_dataset')
if 'features_dataset' not in st.session_state:
    features = pd.read_pickle('features_dataset')
    # Get ItemIDs and ColorIDs from images_dataset, add the features.
    st.session_state.features_dataset = st.session_state.images_dataset.copy()[['ITEMID','COLORID']]
    st.session_state.features_dataset['Feature'] = features

# Initialize models.
if 'detector' not in st.session_state:
    st.session_state.detector = YOLO(config['DetectionModel'])
    target_names = config['DetectionModel'].split(',')
    st.session_state.target_IDs = [key for key,value in st.session_state.detector.names.items()
                                  if value in target_names]
    
if 'preprocessor' not in st.session_state:
    weights = ResNet50_Weights.DEFAULT
    st.session_state.preprocessor = st.session_state.weights.transforms()

if 'feature_extractor' not in st.session_state:
    st.session_state.feature_extractor = resnet50(weights=st.session_state.weights)
    # Replace the final layer with an identity layer,
    # so that it will act as a feature extractor.
    st.session_state.feature_extractor.fc = torch.nn.Identity()
    st.session_state.feature_extractor.eval()

# Initialize variables
if 'bbox_selection' not in st.session_state:
    st.session_state.bbox_selection = None
if 'image_detected' not in st.session_state:
    st.session_state.image_detected = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []


# Build the page layout.

# Divide the page into two columns.
left_column, right_column = st.columns([1,2])
with left_column:
    with st.container(border=False):
        with st.container(border=True):
            # Ask the user to type an image URL.
            url_search = st.text_input('Image URL:')

        image = None
        if url_search:
            try:
                image = image_by_URL(url_search)
            except:
                st.error('URL cannot be opened. Is this an image URL?')
            
        if image:
            with st.container(border=False):
                # Run inference on the image.
                results = st.session_state.detector.predict(source=image,
                                                            verbose=False,
                                                            agnostic_nms=True,
                                                            conf=0.25,
                                                            iou=0.1,
                                                            classes=st.session_state.target_IDs)[0]
                confsModel = results.boxes.conf.numpy()
                bboxesModel = results.boxes.xyxy.numpy()

                if len(bboxesModel) > 0:
                    st.session_state.bbox_selection = None

                    # Show the image with detected objects.
                    # It returns a BGR image, [:,:,::-1] part converts it to RGB.
                    image_drawn = results.plot(conf=len(bboxesModel) > 1)[:,:,::-1]

                    # Ask the user if there are multiple detected objects.
                    if len(bboxesModel) > 1:
                        idx = st.radio('Which detection?',
                                       range(len(confsModel)),
                                       horizontal=True,
                                       format_func= lambda x: round(confsModel[x],2))
                        image_detected = image.crop(bboxesModel[idx])
                    else:
                        image_detected = image.crop(bboxesModel[0])
                    st.image(image_drawn)
                else:
                    # Ash the user to select an area manually.
                    st.error('No detection, select manually:')
                    image_detected = None

                    # image_selector module, as you can except, lets user to select an area of an image.
                    st.session_state.bbox_selection = image_selector(image=image_detected,
                                                                        selection_type='box')
                if st.session_state.bbox_selection:
                    if st.session_state.bbox_selection['selection']['box']:
                        bbox = st.session_state.bbox_selection['selection']['box'][0]
                        image_detected = image.crop(bbox)

with right_column:
    with st.container():
        if url_search and image:
            if not image_detected:
                st.error('No detection or selection!')
            else:
                # Preprocess the image and extract the features.
                image_prep = st.session_state.preprocessor(pad_to_square(image_detected)).unsqueeze(0)
                with torch.no_grad():
                    features = st.session_state.feature_extractor(image_prep).squeeze(0).numpy()
                
                # Find the closest products in the images dataset.
                st.session_state.search_results = find_closest(features,
                                                               st.session_state.features_dataset,
                                                               threshold=0,
                                                               n_matches=20)
                
                # Print the matched products.
                print_all_colors(st.session_state.search_results.ITEMID.tolist(),
                                 st.session_state.images_dataset,
                                img_folder='./temp_images/')