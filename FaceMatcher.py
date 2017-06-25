import os
import numpy as np
from PIL import Image

class FaceMatcher(object):
    def __init__(self, dir):
        file_list = os.listdir(dir)
        self.factory_data = []
        for f in file_list:
            img = Image.open(dir+'/'+f)
            self.factory_data.append(np.array(img).tolist())

    def distance_to(self, from_image, to_image):
        distance = 0
        for idx in range(len(from_image)):
            distance = distance + sum([x - y for x, y in zip(from_image[idx], to_image[idx])])

        return distance / len(from_image)

    def closest_to(self, image):
        smallest_distance = self.distance_to(image, self.factory_data[0])
        closest_image = self.factory_data[0]
        for idx in range(1,len(self.factory_data)):
            current_distance = self.distance_to(image, self.factory_data[idx])
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                closest_image = self.factory_data[idx]

        return closest_image




fm = FaceMatcher('C:/Users/Kanchine/Desktop/Global_AI_Hackathon_2017/test_folder')
target_image = np.array(Image.open('C:/Users/Kanchine/Desktop/1.jpg'))
result_image = fm.closest_to(target_image)

print(1)