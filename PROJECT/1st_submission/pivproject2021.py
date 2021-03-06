import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import os

### AUXILIARY FUNCTIONS ########################################
def get_average(w):
    x = sum([x for (x, y) in w])/len(w)
    y = sum([y for (x, y) in w])/len(w)
    return [int(x), int(y)]
################################################################

def get_corners(path_to_image, expected_IDS=None): #GET COORDINATES FROM ARUCO MARKERS
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
    parameters = cv2.aruco.DetectorParameters_create()

    img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    corners = dict()
    if (expected_IDS):
        success = (len(markerIds)==len(expected_IDS))
    else:
        success = 1
        corners['IDS']=markerIds

    if (success):
        for i in range(len(markerIds)):
            corners[markerIds[i][0]] = [[int(x), int(y)] for [x, y] in markerCorners[i].tolist()[0]]
    else:
        corners = None
    return corners

def get_plane(path_to_image, corners, output_dir, template_points, IDS, template_size, draw_squares=False, scale_down=5):
    '''
    ##-------------------------------------------------------INPUTS------------------------------------------------------------------##
        path_to_image   - address of image to which we want to apply warping;
        corners         - position of the center of each aruco marker. Should come as a dictionary where key is marker ID.
                          Should be output of get_corners;
        output_dir      - address of output directory. If no such directory exists, one will be created;
        template_points - same as corners but for the template image;
        draw_squares    - boolean. If true will save output image as a plot with input image with the Aruco markers highlighted in red
                          on the left side and the respective output image on the right. Useful for troubleshooting;
        template_size   - dimensions of the template image (pixel x pixel);
        scale_down      - scaling factor for produced image. Trades resolution for better performance.
    ##-------------------------------------------------------------------------------------------------------------------------------##
    '''
    assert len(IDS) >= 4, "At least 4 points are needed to compute homography matrix. Undetermined system has infinte solutions."
    frame=path_to_image[-8:-4]

    p1 = [get_average(corners[ID]) for ID in IDS]
    p1 = [[y, x] for [x, y] in p1]
    p2 = [[int(y/scale_down), int(x/scale_down)] for [x, y] in template_points] #SCALE DOWN TEMPLATE IMAGE TO SAVE COMPUTATION TIME

    im = plt.imread(path_to_image).copy() #SOMEHOW, SOME IMAGES ARE IMMUTABLE ARRAYS. ADDED .copy() TO FIX THAT ISSUE
    im = im[:,:,:3] #REMOVES ALPHA CHANNEL (A PROBLEM WITH SOME .PNG FILES)

    if draw_squares:
        for i in range(len(IDS)):
            im[p1[i][0]-10:p1[i][0]+10,p1[i][1]-10:p1[i][1]+10,:] = [250,0,0]

    A = np.zeros((len(IDS)*2,9))

    for i in range(len(IDS)):
            A[i*2,:] = [ p1[i][1], p1[i][0], 1, 0, 0, 0, -p2[i][1]*p1[i][1], -p2[i][1]*p1[i][0], -p2[i][1] ]
            A[i*2+1,:] = [0, 0, 0, p1[i][1], p1[i][0], 1, -p2[i][0]*p1[i][1], -p2[i][0]*p1[i][0], -p2[i][0] ]
    #SVD
    [U,S,V]=np.linalg.svd(A)
    h = V[-1,:]
    #HOMOGRAPHY MATRIX
    H = np.reshape(h,(3,3))
    #DIMENSIONS FOR THE OUTPUT IMAGE
    (x_min, x_max) = (0, int(template_size[0]/scale_down))
    (y_min, y_max) = (0, int(template_size[1]/scale_down))

    transformed = np.zeros((int(y_max),int(x_max),3))
    #WE CORRESPONDED PIXELS IN OUTPUT IMAGE TO PIXELS IN INPUT IMAGE WITH INVERSE H, IN ORDER TO 'FILL' THE BLACK PIXELS THAT WOULD EXIST IF WE USED H ON THE INPUT IMAGE.
    Hi = np.linalg.inv(H) #INVERSE HOMOGRAPHY MATRIX.
    for i in range(transformed.shape[0]):
            for j in range(transformed.shape[1]):
                uv = np.array([[j+x_min],[i+y_min],[1]])
                xy = np.dot(Hi,uv)
                x1=int(xy[0]/xy[2])
                y1=int(xy[1]/xy[2])
                if x1>0 and y1>0 and y1<len(im) and x1<len(im[0]): #FIXES "OUT OF BOUNDS" INDEX ERROR
                    #print(im[y1,x1,:])
                    transformed[i,j,:] = im[y1,x1,:]/260 #NORMALIZATION NEEDED TO SUCESSFULY SAVE IMAGE.
                    #SOME IMAGES HAD PIXELS WITH MORE THAN 250 IN ONE OF THE CHANNELS

    #OUTPUTS PLOT WITH ORIGINAL IMAGE ON THE LEFT AND OUTPUT IMAGE ON THE RIGHT
    if (draw_squares):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.imshow(im)
        ax2.imshow(transformed)
        fig.savefig(output_dir+'/'+frame+'.jpg')
        plt.close('all')
    else:
        plt.imsave(output_dir+'/'+frame+'.jpg', transformed)

def main():
    user_input = sys.argv

    task       = user_input[0]
    input_dir  = user_input[4]
    output_dir = user_input[3]
    template   = user_input[2]
    image_files= sorted(os.listdir(input_dir))

    #CHECK IF OUTPUT FILE EXISTS. IF NOT, CREATE DIRECTORY
    if (not os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    tmp_corners     = get_corners(template)
    IDS             = [int(id) for id in tmp_corners['IDS']]
    coords_template = [get_average(tmp_corners[id]) for id in IDS]
    template_image  = plt.imread(template)
    template_size   = (len(template_image[0]), len(template_image))

    count=1
    for image_file in image_files:
        print ("\rProcessing image: " + str(count) + " of " + str(len(image_files)) + ' filename: ' + image_file, end='')
        path_to_image=input_dir+'/'+image_file
        if (os.path.isfile(path_to_image)):
            new_corners = get_corners(path_to_image, coords_template)
            if (new_corners): #IF ARUCO MARKERS ARE NOT DETECTED, WE USE THE COORDINATES FROM THE PREVIOUS FRAME
                corners = new_corners
            get_plane(path_to_image=path_to_image, corners=corners, template_points=coords_template, output_dir=output_dir, template_size=template_size, IDS=IDS)
        else:
            print(path_to_image + 'is empty. Closing program')
            return 1
        count+=1
    print ('\rExecution Successful!                                        ')

if __name__ == '__main__':
    main()
