from cgi import print_arguments
import cv2 as cv
import numpy as np
import time

# img_path = "data/palm.jpeg"
# img = cv.imread(img_path) #read the image
# cv.imshow('palm image',img)
# cv.waitKey()

def skin_mask(img):
    hsv_image = cv.cvtColor(img,cv.COLOR_BGR2HSV) # convert the rgb image to hsv
    lower = np.array([0, 48, 80], dtype = "uint8") # lower and upper hsv values for skin
    upper = np.array([20, 255, 255], dtype = "uint8")
    skin_region = cv.inRange(hsv_image,lower,upper) #get the skin coloured region
    blurred = cv.blur(skin_region, (2,2)) #blur to correctly get external contour
    ret,thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
    cv.imshow("thresh", thresh)
    cv.waitKey()
    return thresh

def draw_contours(thresh,img):
    contours,hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) #obtain contours from thresh
    # contours = max(contours, key=lambda x: cv.contourArea(x))
    cv.drawContours(img,contours,-1,(255,255,0),2)
    # cv.imshow("contours",img)
    # cv.waitKey()
    return contours

def convex_hull(contours,img):
    contour = max(contours, key=lambda x: cv.contourArea(x)) #find the contour with max area
    hull = cv.convexHull(contour,returnPoints=False)
    # cv.drawContours(img, [hull], -1, (0, 255, 255), 2)
    # cv.imshow("hull", img)
    # cv.waitKey()
    return hull

def calculate_fingers_from_convexity_defects(contours,hull,img):
    contour = max(contours, key=lambda x: cv.contourArea(x)) #find the contour with max area
    defects =cv.convexityDefects(contour,hull)
    count=0
    if defects is not None:
        for i in range(defects.shape[0]):
            start_point,end_point,farthest_point,distance =defects[i][0] # obtain the start etc position of the point in the contour
            start = tuple(contour[start_point][0]) # find the acutal cordinates of the points
            end = tuple(contour[end_point][0])
            far = tuple(contour[farthest_point][0])

            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) #get the length of the sides
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) # cosine theorem

            if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
                count += 1  #increase the number of fingers
                cv.circle(img, far, 4, [0, 0, 255], -1) # draw a circle showing the farthest point

    if count > 0:
        count = count+1
    cv.putText(img, str(count), (0, 50), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv.LINE_AA)
    # cv.imshow("hull", img)
    # cv.waitKey()


if __name__=="__main__":
    img_path = "data/twome.jpg"
    img = cv.imread(img_path) 
    thresh =skin_mask(img)
    contours=draw_contours(thresh,img)
    hull= convex_hull(contours,img)
    calculate_fingers_from_convexity_defects(contours,hull,img)
