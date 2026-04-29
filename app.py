import streamlit as st
from PIL import Image
import pickle
import cv2
import os
from mtcnn import MTCNN
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from tensorflow.keras.layers import GlobalMaxPool2D
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity


# Upload the image paths and extracted features 
feature_list = pickle.load(open('features.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

def save_uploaded_image(uploaded_img):
    try:
        with open(os.path.join('uploads',uploaded_img.name),'wb') as f:
            f.write(uploaded_img.getbuffer())
        return 1
    except:
        return 0

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

def recommend(sample_img_feature):
    similarity =[]
    for i, arr in enumerate(feature_list):
        simi = cosine_similarity(sample_img_feature.reshape(1,-1),arr.reshape(1,-1))[0][0]
        similarity.append(simi)
    
    index_of_max = similarity.index(max(similarity))
    predicted_path = filenames[index_of_max]
    predicted_chance = f"{(float(max(similarity))*100):.2f}%"

    return predicted_path, predicted_chance

def face_detect(img_path):
    to_face_img = cv2.imread(img_path)
    detector = MTCNN()
    for i in range(10):
        if to_face_img.shape[0]>800 and to_face_img.shape[1]>800:
            to_face_img = cv2.resize(to_face_img, (int((to_face_img.shape[1])/2),int((to_face_img.shape[0])/2)))
        else: 
            break
            
    # Detect Face to extract features
    result = detector.detect_faces(to_face_img)
    x,y,width,height = result[0]['box']
    face = to_face_img[y:y+height,x:x+width]
 
    # Extracting features for comparision
    face1 = cv2.resize(face, (224,224))
    return face1

st.title('Which Bollywood Celebraty Are You?')

uploaded_file = st.file_uploader('Choose an Image', type=["jpg", "jpeg", "png"])

if uploaded_file:
    #save the imgae in directory
    if save_uploaded_image(uploaded_file):
        #Load image
        st.success('Successfully uploaded the image, Please wait', icon="✅")
        col1,col2 = st.columns(2)
        display_image = Image.open(uploaded_file)
        with col1:
            st.image(display_image, caption='Image uploaded successfully')

        face = face_detect(os.path.join('uploads',uploaded_file.name))
        feature = extract_features(face)
        predicted_path, predicted_chance = recommend(feature)

        with col2:
            st.image(predicted_path, caption=f"{predicted_path.split('/')[1].replace('_', ' ')} — Chances: {predicted_chance}")

    else:
        st.error("Image is not uploading, please upload in either jpg, jpeg, png", icon="🚨")

        
        




















        