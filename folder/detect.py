import numpy as np
import pandas as pd


def bbox_iou(bboxA, bboxB):
    """Compute Intersection of Union of two bounding boxes."""

    # Determine the (x, y) coordinates of the intersection rectangle.
    x_min = max(bboxA[0], bboxB[0])
    y_min = max(bboxA[1], bboxB[1])
    x_max = min(bboxA[2], bboxB[2])
    y_max = min(bboxA[3], bboxB[3])
    # Compute the area of intersection rectangle.
    area_inter = max(0, x_max-x_min+1) * max(0, y_max-y_min+1)
    # Conpute the area of both bounding boxes.
    area_bboxA = (bboxA[2]-bboxA[0]+1) * (bboxA[3]-bboxA[1]+1)
    area_bboxB = (bboxB[2]-bboxB[0]+1) * (bboxB[3]-bboxB[1]+1)
	# Compute the area of union.
    area_union = float(area_bboxA + area_bboxB - area_inter)
    return area_inter / area_union


def cosine_similarity(arrayA: np.ndarray,
                      arrayB: np.ndarray) -> np.ndarray:
    """
    Calculate the cosine similarity between corresponding vectors in two matrices.
    
    Args:
    arrayA: The first matrix, where each row is a vector.
    arrayB: The second matrix, where each row is a vector.
    
    Returns:
    A numpy array containing the cosine similarity scores for each pair of corresponding vectors.
    """
    arrayA_2D = np.array(arrayA).reshape((-1,arrayA.shape[-1]))
    dot_products = np.dot(arrayA_2D, np.array(arrayB).T)
    norm_arrayA = np.linalg.norm(arrayA_2D, axis=1,keepdims=True)
    norm_arrayB = np.linalg.norm(np.array(arrayB), axis=1)
    
    similarities = dot_products / (norm_arrayA * norm_arrayB)
    
    # Handle cases where norm is zero
    similarities[np.isnan(similarities)] = 0
    
    return similarities


def find_closest(feature_image: list[np.ndarray],
                 df_features: pd.DataFrame,
                 threshold: 0.5,
                 n_matches=5):
    """
    Determine the closest products of given features of image(s).
    If multiple images are given, they are assumed to be images of the same object,
    and the scores are aggregated by their mean.

    feature_image: an array or list of arrays (multiple images) containing features.
    df_features: DataFrame of features of products that will be tested against input.
    threshold: the minimum required score to accept if images are similar.
    n_matches: Number of results to return.
    """

    if isinstance(feature_image, np.ndarray):
        X = [feature_image]
    elif isinstance(feature_image, list[np.ndarray]):
        X = feature_image.copy()
    else:
        raise TypeError('feature_image must be numpy array or a list of arrays')

    df_distances_result = df_features.copy()
    for col in ['COLORID','Feature']:
        try:
            df_distances_result = df_distances_result.drop(col,axis=1)
        except:
            continue
    
    scores = cosine_similarity(np.array(df_features.Feature.to_list()), X)

    # If there are multiple images, seperate each score of corresponding images.
    if np.shape(X)[0] > 1:
        mask = np.any(scores > threshold,axis=-1)
        for i in range(np.shape(X)[0]):
            df_distances_result.insert(len(df_distances_result.columns),
                                       f'Score_{i}',
                                       scores[:,i])
        
        # Group by ItemIDs and select the best score of each ItemID.
        df_distances_result = df_distances_result[mask].groupby('ITEMID',as_index=False).agg('max')

        # Take the mean of scores for images.
        df_distances_result['Score'] = df_distances_result[['Score_'+ str(i) for i in range(np.shape(X)[0])]].mean(axis=1)
        df_distances_result = df_distances_result.drop(['Score_'+ str(i) for i in range(np.shape(X)[0])],axis=1)
    else:
        mask = scores > threshold
        df_distances_result.insert(len(df_distances_result.columns),'Score',scores)
        # Group by ItemIDs and select the best score of each ItemID.
        df_distances_result = df_distances_result[mask].groupby('ITEMID',as_index=False).agg('max')
	
    # Sort by highest score, return only n_matches results.
    return df_distances_result[['Score','ITEMID']].sort_values('Score',ascending=False).iloc[:n_matches]