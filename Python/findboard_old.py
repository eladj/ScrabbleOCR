# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:20:14 2015

@author: elad
"""
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pytesseract


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "int32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def order_points2(pts):
    """  Order the corners from top-left clockwise """
    # pts - numpy array of 4x2    
    mu = cv2.moments(contours[0], False)
    mass_center =  (mu['m10']/mu['m00'] , mu['m01']/mu['m00'] )    
    top, bottom = [], []
    for n in range(4):
        if pts[n,1] < mass_center[1]: # y-axis is downward
            top.append(pts[n,:])
        else:
            bottom.append(pts[n,:])
    tl = top[1] if top[0][0] > top[1][0] else top[0]
    tr = top[0] if top[0][0] > top[1][0] else top[1]
    bl = bottom[1] if bottom[0][0]> bottom[1][0] else bottom[0]
    br = bottom[0] if bottom[0][0] > bottom[1][0] else bottom[1]        
    corners = np.zeros((4,2)).astype(np.int32)
    corners[0,:] = tl
    corners[1,:] = tr
    corners[2,:] = br
    corners[3,:] = bl
    return corners

def four_point_transform(image, pts, dst=None):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    if dst == None:
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
    
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
    else:
        maxWidth, maxHeight = dst

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order        
    dst = np.array([ [0, 0], [maxWidth - 1, 0],
                     [maxWidth - 1, maxHeight - 1],
                     [0, maxHeight - 1]], dtype = "float32")        
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

if __name__ == "__main__":
    #img_fn = r"C:\Users\elad\Documents\code\DigitalScrabble\board_OnePlus (1).jpg"
    #img_fn = r"C:\Users\elad\Documents\code\DigitalScrabble\board_letters (3).jpg"
    img_fn = r"C:\Users\elad\Desktop\IMG_BOARD.jpg"
    #img_fn = r"C:\Users\elad\Documents\code\DigitalScrabble\board_nexus3 (3).jpg"
    
    im_size = 8e6 #in total pixels. The size to set the image (larger will shrink and smaller will enlarge)
    blur_size = (5,5)
    blur_std = 5
    open_close_kernel_size = (10, 10)
    curve_approx_eps = 15 # maximum distance between the original curve and its approximation
    warped_shape = (1024, 1024) # to which shape wrap the board
    grid_size = (8,8) # x,y
    border_shift = 55 #pixels. from outer border to inner
    tile_std_th = 10 # STD of each tile Hue, to decide if it is occupied or not
    letter_bw_th = 150 # threshold to seperate tile's letter from background
    
    #%%
    bgr = cv2.imread(img_fn)
    # Bring all images to the same size
    factor = np.round(np.sqrt(im_size/(bgr.shape[0]*bgr.shape[1])),2)
    if factor < 1: interpolation = cv2.INTER_AREA  #shrink
    else: interpolation = cv2.INTER_LINEAR #enlarge
    bgr = cv2.resize(bgr,None, fx=factor, fy=factor)    
    rgb = cv2.cvtColor(bgr.copy(), cv2.COLOR_BGR2RGB)
    rgb = cv2.GaussianBlur(rgb, blur_size, blur_std)
    
    rgbPyrDown = cv2.pyrDown(rgb)
    rgbPyrDown = cv2.pyrDown(rgbPyrDown) # Downsample image by 4
    r,g,b = rgbPyrDown[:,:,0],rgbPyrDown[:,:,1],rgbPyrDown[:,:,2]
    hsv = cv2.cvtColor(rgbPyrDown.copy(), cv2.COLOR_RGB2HSV)
    h,s,v = hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    
    #%% Thresholding    
    lower_red = (0, 50, 50)
    upper_red = (9, 230, 235)
    bw = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = (170, 50, 50)
    upper_red = (180, 230, 235)
    bw2 = cv2.inRange(hsv, lower_red, upper_red)
    bw = np.uint8(np.logical_or(bw,bw2))
    kernel = np.ones(open_close_kernel_size ,np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel) # opening (remove small objects from the foreground)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel) # closing (fill small holes in the foreground)
    
    #%% Find Contour and 4 Corners
    bwCanny = cv2.Canny(bw, 1, 1)

    
    #%%
    image, contours, hierarchy = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
    rgb_contours = rgb.copy()
    rgb_contours_approx = rgb.copy()
    rgb_warped = None
    if contours != []:
        for contour in contours:
            if np.abs(cv2.contourArea(contour)) < 15000: 
                continue
            contour = contour*4 # Upsample back to original image size
            epsilon = 15 # maximum distance between the original curve and its approximation
            approxCurve = cv2.approxPolyDP(contour, curve_approx_eps, True)    
            corners = approxCurve.reshape((4,-1))
            #minRect = cv2.minAreaRect(contour)
            #rectPoints = cv2.boxPoints(minRect).astype(np.int32)
            rgb_contours_approx = rgb.copy()
            cv2.drawContours(rgb_contours, contour, 0, (255,255,0), 5)
            #cv2.drawContours(rgb_contours_approx, rectPoints.reshape((4,-1,2)), 0, (255,255,0), 5)
            
            # Arrange orners by order
            #corners = rectPoints.reshape((4,-1))            
            corners = order_points(corners)
            colors = ((255,0,0), (0,255,0), (0,0,255), (255,255,255))
            for n in range(4):
                cv2.circle(rgb_contours_approx, tuple(corners[n,:].tolist()), 35, colors[n],-1)
            # Apply the perspective transformation
            rgb_warped = four_point_transform(rgb.copy(), corners, warped_shape)
    #%% Build Tiles grid
    rgb_warped_plot = rgb_warped.copy()
    tile_width, tile_height = (np.array(warped_shape) - 2*border_shift) / np.array(grid_size)
    origin = np.array([border_shift + tile_width/2, border_shift + tile_height/2])
    grid = np.zeros((grid_size[0], grid_size[1],2), dtype="int32") # x,y, pixel x,y
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            grid[i,j,:] = np.array([origin[0] + tile_width*j,
                                    origin[1] + tile_height*i]).astype("int32")
            cv2.circle(rgb_warped_plot, tuple(grid[i,j,:].tolist()), 35, (0,255,255),-1)
            cv2.putText(rgb_warped_plot, "%d,%d" % (i,j), tuple(grid[i,j,:].tolist()),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,255),2 ,2)
    #%% Check if grid occupied
    hsv2 = cv2.cvtColor(rgb_warped.copy(), cv2.COLOR_RGB2HSV)
    h2,s2,v2 = hsv2[:,:,0],hsv2[:,:,1],hsv2[:,:,2]
    occupied_tiles = []
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):            
            x,y = grid[i,j,:]
            tile_roi = h2[y-tile_height/2+20:y+tile_height/2-20,
                          x-tile_width/2+20:x+tile_width/2-20]
            tile_std = np.std(tile_roi)
            #print("i=%d, j=%d, std=%.2f" % (i,j,tile_std))
            if tile_std > tile_std_th:
                occupied_tiles.append((i,j))
                cv2.circle(rgb_warped_plot, tuple(grid[i,j,:].tolist()), 30, (255,255,0),-1)                        
    #%% Build Lettes Dict
    rgb_letters_plots = rgb_warped.copy()
    letters = [] 
    for tile_ij in occupied_tiles:
        letter = {}
        i,j = tile_ij        
        x,y = grid[i,j,:]
        tile_roi = v2[y-tile_height/2+25:y+tile_height/2-25,
                      x-tile_width/2+25:x+tile_width/2-25]
                      
        tile_bw = tile_roi > letter_bw_th
        pil_img = Image.fromarray(np.uint8(tile_bw))
        tile_ocr = pytesseract.image_to_string(pil_img, config="-psm 10")
        letter['i'], letter['j'] = i,j                
        letter['bw'] = tile_bw
        letter['ocr'] = tile_ocr
        letters.append(letter)
        print("i=%d, j=%d, OCR=%s" % (i,j, tile_ocr))
        cv2.putText(rgb_letters_plots, "%s" % tile_ocr, tuple((grid[i,j,:]-4).tolist()),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 3 ,2)    
        cv2.putText(rgb_letters_plots, "%s" % tile_ocr, tuple(grid[i,j,:].tolist()),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,0), 3 ,2)                        
    
    
    #       
    #minLineLength = 100
    #maxLineGap = 1
    #lines = cv2.HoughLinesP(bw.copy(), 1, np.pi/180, 100, minLineLength, maxLineGap)
    #rgb_hough_lines = rgb.copy()
    #for x1,y1,x2,y2 in lines[:,0,:]:
    #    cv2.line(rgb_hough_lines,(x1,y1),(x2,y2),(0,255,0),2)
    
    #%% Plot
    # Plot RGB and HSV
    fig = plt.figure()
    ax1 = fig.add_subplot(2,3,1)
    ax1.imshow(r, cmap='gray')
    ax1.set_title("Red")
    ax1.format_coord = lambda x,y: "x=%.1f, y=%.1f, Red=%1.f" % (x, y, r[int(y),int(x)])
    ax2 = fig.add_subplot(2,3,2)
    ax2.imshow(g, cmap='gray')
    ax2.set_title("Green")
    ax2.format_coord = lambda x,y: "x=%.1f, y=%.1f, Green=%1.f" % (x, y, g[int(y),int(x)])
    ax3 = fig.add_subplot(2,3,3)
    ax3.imshow(b, cmap='gray')
    ax3.set_title("Blue")
    ax3.format_coord = lambda x,y: "x=%.1f, y=%.1f, Blue=%1.f" % (x, y, b[int(y),int(x)])
    ax4 = fig.add_subplot(2,3,4)
    ax4.imshow(h, cmap='gray')
    ax4.set_title("Hue")
    ax4.format_coord = lambda x,y: "x=%.1f, y=%.1f, Hue=%1.f" % (x, y, h[int(y),int(x)])
    ax5 = fig.add_subplot(2,3,5)
    ax5.imshow(s, cmap='gray')
    ax5.set_title("Saturation")
    ax5.format_coord = lambda x,y: "x=%.1f, y=%.1f, Saturation=%1.f" % (x, y, s[int(y),int(x)])
    ax6 = fig.add_subplot(2,3,6)
    ax6.imshow(v, cmap='gray')
    ax6.set_title("Value")
    ax6.format_coord = lambda x,y: "x=%.1f, y=%.1f, Value=%1.f" % (x, y, v[int(y),int(x)])
    # Plot Threshold
    fig2 = plt.figure()
    ax1_2 = fig2.add_subplot(2,2,1)
    ax1_2.imshow(rgb)
    ax1_2.set_title("RGB")
    ax2_2 = fig2.add_subplot(2,2,2)
    ax2_2.imshow(bw, cmap='gray')
    ax2_2.set_title("BW")
    ax3_2 = fig2.add_subplot(2,2,3)
    ax3_2.imshow(rgb_contours_approx)
    ax3_2.set_title("Contours Approximate Lines")
    ax4_2 = fig2.add_subplot(2,2,4)
    ax4_2.imshow(rgb_warped)
    ax4_2.set_title("RGB Warped")    
    # Plot Grid
    fig3 = plt.figure()
    ax1_3 = fig3.add_subplot(2,2,1)
    ax1_3.imshow(rgb_warped_plot)
    ax1_3.set_title("Grid Detection")
    ax2_3 = fig3.add_subplot(2,2,2)
    ax2_3.imshow(rgb_letters_plots)
    ax2_3.set_title("Letters OCR")
    
    """
    HSV color space is also consists of 3 matrices, HUE, SATURATION and VALUE.
    In OpenCV, value range for  HUE, SATURATION  and VALUE are 
    respectively 0-179, 0-255 and 0-255.
    HUE represents the color, SATURATION represents the amount to which that
    respective color is mixed with white and VALUE represents the  amount to 
    which that respective color is mixed with black.
    
    red object has HUE, SATURATION and VALUE in between 170-180, 160-255, 60-255 
    
    Hue values of basic colors
        Orange  0-22
        Yellow 22- 38
        Green 38-75
        Blue 75-130
        Violet 130-160
        Red 160-179
    """