
#===================== Set up ===================================================
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from cv2 import imread, resize, imwrite, cvtColor, COLOR_BGR2RGB, COLOR_BGR2GRAY, IMREAD_UNCHANGED
from skimage.metrics import structural_similarity as ssim

# import labels for the reference images
labels = pd.read_csv('./labels.csv',index_col = 0)

#===================== Functions =================================================
def crop_sign(img):
    """
    Crop the sign from the photo.
    Use the mask to filter only the yellow color, apply some threshold.
    Contour the yellow area, crop those area, save img as file
    """
    # prepare images
    rgb = cvtColor(img, COLOR_BGR2RGB)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # filter only the yellow color
    mask1= cv2.inRange(hsv, np.array([120,255,255]), np.array([120,255,255]))
    mask2 = cv2.inRange(hsv, np.array([10,120,120]), np.array([30,255,255]))
    result_yellow = cv2.bitwise_and(rgb, rgb, mask = mask1+mask2)
    gray_new = cv2.cvtColor(result_yellow,cv2.COLOR_BGR2GRAY)

    # apply a bit of threshold to confirm the yellow part
    ret, bw = cv2.threshold(gray_new, 65, 255, cv2.THRESH_BINARY)

    # find the yellow area, crop 
    contours, hiera = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img2 = img.copy()
    
    store = 0
    for i in range(len(contours)):
      cnt_area = cv2.contourArea(contours[i])
      if cnt_area > 2000:
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = img2[y:y+h, x:x+w]
        cropped = cvtColor(cropped, COLOR_BGR2RGB)
        store = np.array(cropped)

    return store

def get_resize_image(row_id, path):
    """
    Read the image given file name and resize them. Returns numpy array.
    """
    # read the image
    file_path = path + str(row_id) + ".jpg"
    print(file_path)
    img = imread(file_path, IMREAD_UNCHANGED)

    # resize and convert to grayscale
    resize_img = cv2.resize(img,(255,255))
    squ_img = np.squeeze(resize_img)
    gray = cv2.cvtColor(squ_img, cv2.COLOR_BGR2GRAY)
    
    return np.array(gray)

def mse(imageA,imageB):
    """
    Calculate MSE value. Returns MSE.
    """
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

def compare_images(imageA,imageB):
    """
    Apply threshold method and use MSE,SSIM to compare images. Return m,s values.
    MSE the lesser, the better.
    SSIM the higher, the better.
    """
    # apply threshold to strengthen yellow part 
    ret, bwA = cv2.threshold(imageA, 60, 255, cv2.THRESH_BINARY) 
    ret, bwB = cv2.threshold(imageB, 60, 255, cv2.THRESH_BINARY) 

    # calculate mse and ssim values 
    m = mse(bwA,bwB)
    s = ssim(bwA,bwB)
    print("MSE: {:.2f},SSIM: {:.2f}".format(m,s))
    
   
    return m,s,imageB

def choose_ref(list):
    """
    Calling a list: (mse,ssim,ref_photo) and calculate the max,min,mean
    Select the best ref photo based on the measurement
    Return the labels
    """
    data = pd.DataFrame(list)
    min_m = data[0].min()
    max_s = data[1].max()
    mean_s = data[1].mean()
    print("min mse:",min_m)
    print("max ssim:",max_s)
    print("mean ssim:",mean_s)
    
    # loop over the list and choose the reference images based on mse and ssim values
    for i in range(len(list)):
      if list[i][0] == min_m and (list[i][1] <= max_s and list[i][1] > mean_s):
        img = np.array(list[i][2])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(i)

        return i+1

#===================== WEB ===================================================       
# setup page
CURRENT_THEME = "dark"
IS_DARK_THEME = True
st.set_page_config(page_title='Sign Detection', page_icon="❄")
st.set_option('deprecation.showfileUploaderEncoding', False) # disable deprecation error
with open("app.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# program
#Add sidebar
st.sidebar.title("Yak Pen Spy Der Man :spider:")
st.sidebar.markdown("### Group Member")
st.sidebar.markdown("Chananyu Kamolsuntron")
st.sidebar.markdown("Nonthakorn Chencharatmatha")
st.sidebar.markdown("Nithiwat Pattrapong")

#Add title and subtitle
st.title(" ")
st.title("Traffic Sign Detector :wave:")
st.markdown("What is the meaning of traffic signs on the road?")
st.markdown("Upload the picture of that sign then we will translate it for you.")

img = st.file_uploader("Please upload the file", type=["jpg"])
if img is not None:
    img = Image.open(img).convert('RGB')  #PIL syntax
    new_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # convert PIL obj to cv2 obj
    sign = crop_sign(new_img)
    
    col1, col2, col3, col4, col5 = st.columns([1,3,1,3,1])
    with col1:
        st.write('')
    with col2:
        st.image(img,"Original image")
    with col3:
        st.write('➜')
    with col4:
        st.image(sign,"Extracted sign")
    with col5:
        st.write('')
    
    
    # choose ref
    sign = cv2.resize(sign,(255,255))
    re_sign = np.squeeze(sign)
    gray = cv2.cvtColor(re_sign, cv2.COLOR_BGR2GRAY)
    values = []
    j = 1
    while j <= 15:
        ref = get_resize_image(j,"/content/drive/MyDrive/spyder photo/ref/")
        values.append(compare_images(gray,ref))
        j+=1
    
    index = choose_ref(values)
    col1, col2, col3= st.columns([1,3,1])
    with col1:
        st.write('')
    with col2:
        st.success("The meaning of this sign is " + labels['meaning'].loc[index] + ".")
    with col3:
        st.write('')
    

    
    
