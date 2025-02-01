import streamlit as st

import pandas as pd
from PIL import Image

from utils.image import image_by_URL, get_non_white_bounding_box, concat_images_h

def print_all_colors(itemIDs: list[str],
                     image_df: pd.DataFrame,
                     img_folder: str):
    """
    Print the images of products for all available color options.
    
    image_df: The DataFrame containing the images.
    It is expected to have columns ['ITEMID','COLORID','URL']

    img_folder: The folder to save the resultant images.
    Saves time if the images will be printed later.
    """

    for itemID in itemIDs:
        st.divider()
        st.write(itemID)

        # Filter the DataFrame by ItemID.
        df_itemid = image_df[image_df.ITEMID == itemID]
        
        # Loop through all colors.
        for colorid in df_itemid.COLORID.unique():
            img_path = f'{img_folder}{itemID}-{colorid}.jpg'
            try:
                # Open the image if exists.
                final_img = Image.open(img_path)
            except:
                imgs = []
                
                for row in df_itemid[df_itemid.COLORID == colorid].iterrows():
                    try:
                        imURL = image_by_URL(row[1].URL)
                        imgs.append(imURL.resize((imURL.size[0]//2,imURL.size[1]//2)))
                    except:
                        continue
                
                # Concat the images horizantally and save.
                final_img = concat_images_h(imgs)
                final_img = final_img.crop(get_non_white_bounding_box(final_img))
                final_img.save(img_path)
            
            # Show the image.            
            st.image(final_img)