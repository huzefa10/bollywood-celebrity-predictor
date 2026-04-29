import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
from deepface import DeepFace
from tensorflow.keras.layers import GlobalMaxPool2D
from numpy.linalg import norm


filenames = pickle.load(open('filenames.pkl', 'rb'))

def extract_features(img_paths):
    deepfaced = DeepFace.represent(img_path=img_paths, model_name='VGG-Face', enforce_detection=False)
    into_arr = np.array(deepfaced[0]['embedding'])
    into_arr_4d = into_arr.reshape(2,2,1024)
    ex_into_arr_4d = np.expand_dims(into_arr_4d,1)
    layer = GlobalMaxPool2D()(ex_into_arr_4d)
    layer_numpy = layer.numpy()
    flat_layer = layer_numpy.flatten()
    norm_layer = flat_layer/norm(flat_layer)
    return norm_layer

features = []
for file in tqdm(filenames):
    features.append(extract_features(file))

pickle.dump(features, open('features.pkl', 'wb'))