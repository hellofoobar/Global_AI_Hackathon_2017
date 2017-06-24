from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc

#Q2:
def MakePyramid(image,minsize):
    # set input image to img
    img = image
    # decleare a list to store  a scaled representation of the input image
    pyramid = []
    # add the input image
    pyramid.append(img)
    # width & height of the image
    width, height = img.size

    while ((width >= minsize) & (height >= minsize)):
    	
    	pastImg = pyramid[-1]
    	width,height = pastImg.size

    	# scaled the input image by giving formula 
    	img = pastImg.resize((int(width*0.75),int(height*0.75)), Image.BICUBIC)

    	width,height = img.size

    	# if the new created image's width and height are greater than minsize
    	# add the image to the list
    	if((width >= minsize) & (height >= minsize)):
            pyramid.append(img)

    return pyramid



# question 3  
def ShowPyramid(pyramid):
    # width is the original image's width.
    # initial the maxHeight to 0
    maxWidth = pyramid[0].size[0]
    maxHeight = 0;
    
    # maxHeight is the sum of all image's height in pyrmaid
    for img in pyramid:
        height = img.size[1]
        maxHeight = maxHeight + height
        
    # create the white image to paste all inamges in pyramid
    image = Image.new("L", (maxWidth, maxHeight),"white")

    # placing pyramid vertically
    offset_y = 0
    for img in pyramid:
        image.paste(img,(0,offset_y))
        offset_y = offset_y + img.size[1]

    image.show()


#Q4:
def FindTemplate(pyramid, template, threshold):
    # create a new template with width = 15
    width = 15
    r = template.size[0] / width
    height = template.size[1] / r
    newTemplate = template.resize((width,height),Image.BICUBIC)

    pixelPositions = []
    for img in pyramid:
        result = ncc.normxcorr2D(img, newTemplate)
        # correlation is bigger than the threshold
        pixelBiggerThreshold = np.where (result > threshold)
        
        # pixelBiggerThreshold contains two elements, for each x and y
        # [0] holds y indices, [1] holds x indices
        c = zip(pixelBiggerThreshold[0],pixelBiggerThreshold[1])
        
        # pixelPositions contains all matching points for each image
        pixelPositions.append(c)
    
    # make image support RBG color, then we can draw the red retangele on the image
    image = pyramid[0].convert('RGB')
    
    for index in range(len(pixelPositions)):
        points = pixelPositions[index]
        # changes in x, width and y, height
        changeInX = newTemplate.size[0] 
        changeInY = newTemplate.size[1] 
        
        # draw red rectangle around the centred of each point
        for point in points:
            # rescale the poins by the right power of 0.75
            xP = point[1] // 0.75 ** index 
            yP = point[0] // 0.75 ** index
            
            # calculate 4 points around the centre
            x2 = xP + changeInX
            x1 = xP - changeInX
            y1 = yP - changeInY
            y2 = yP + changeInY
            
            # draw all 4 lines
            draw = ImageDraw.Draw(image)
       	    draw.line((x1,y1,x2,y1),fill="red",width=2)
       	    draw.line((x1,y2,x2,y2),fill="red",width=2)
       	    draw.line((x1,y1,x1,y2),fill="red",width=2)
       	    draw.line((x2,y1,x2,y2),fill="red",width=2)
            del draw
    
    image.show()
    


#Q5:
# when threshold = 0.575.
# judybats.jpg
# - non-faces: 					1
# - missed faces: 				1 
# students.jpg
# - non-faces: 					3 
# - missed faces: 				4 
# tree.jpg
# - non-faces:    				1
# - missed faces:               0
# family.jpg
# - non-faces:				    0
# - missed faces:               0
# fans.jpg
# - non-faces:				    3
# - missed faces:               3
# sports.jpg
# - non-faces:    				0
# - missed faces:               1
# Error rate = (1+3+1+3) - (1+4+2+1) = 0

#call main() for test
def main():
    img = Image.open('judybats.jpg')
    #img = Image.open('students.jpg')
    #img = Image.open('tree.jpg')
    #img = Image.open('family.jpg')
    #img = Image.open('fans.jpg')
    img = Image.open('sports.jpg')
    minsize = 30
    pyramid = MakePyramid(img, minsize)
    #ShowPyramid(pyramid)
    template = Image.open('template.jpg')
    FindTemplate(pyramid,template,0.575)


# #test and get the output image
main()

#Q6:
'''
# judybats.jpg
# recall rate = 4/5

# students.jpg
# recall rate = 23/27

# tree.jpg
# recall rate = N/A

# family.jpg
# recall rate = 3/3 = 1

# fans.jpg
# recall rate = N/A

# sports.jpg
# recall rate = 0/1 = 0

because we don't have right termplate for NCC, so for some images has very
low recall rate on some image, such as tree.jpg, fans.jpg, sports.jpg
'''

