# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 21:43:01 2017

@author: Seagle
"""
#%%
import face_recognition as fr

image = fr.load_image_file("C:/Users/Seagle/Google Drive/Google Photos/IMG_20160605_102843.jpg")
face_locations = fr.face_locations(image)

#%%
from PIL import Image
top, right, bottom, left = face_locations[0]
face_image = image[top:bottom, left:right]
pil_image = Image.fromarray(face_image)
pil_image.show()