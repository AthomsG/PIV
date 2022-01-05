import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def get_plane(path_to_image, coords_OG, coords_BEV, output_dir, draw_squares=False):

    p2 = coords_OG
    p1 = coords_BEV

    im = plt.imread(path_to_image)

    if (draw_squares):
        #IF WE WANT TO DRAW THE SQUARES
        for i in range(4):
                im[p1[i][0]-2:p1[i][0]+3,p1[i][1]-2:p1[i][1]+3,:] = [255,0,0]
                im[p2[i][0]-2:p2[i][0]+3,p2[i][1]-2:p2[i][1]+3,:] = [0,255,0]

    A = np.zeros((8,9))
    # Homography matrix
    for i in range(4): # Using the corners
            A[i*2,:] = [ p1[i][1], p1[i][0], 1, 0, 0, 0, -p2[i][1]*p1[i][1], -p2[i][1]*p1[i][0], -p2[i][1] ]
            A[i*2+1,:] = [0, 0, 0, p1[i][1], p1[i][0], 1, -p2[i][0]*p1[i][1], -p2[i][0]*p1[i][0], -p2[i][0] ]

    [U,S,V]=np.linalg.svd(A)
    h = V[-1,:]
    H = np.reshape(h,(3,3))

    #THESE WEIRD COORDINATE SWITCHES ARE FREAKING ME OUT
    refX1 = int(np.min([y for (x, y) in p2]))
    refX2 = int(np.max([y for (x, y) in p2]))
    refY1 = int(np.min([x for (x, y) in p2]))
    refY2 = int(np.max([x for (x, y) in p2]))

    #CHOOSES SIZE OF OUTPUT IMAGE
    transformed = np.zeros((int(refY2-refY1),int(refX2-refX1),3))

    Hi = np.linalg.inv(H)
    for i in range(transformed.shape[0]):
            for j in range(transformed.shape[1]):
                        tt = np.array([[j+refX1],[i+refY1],[1]])
                        tmp = np.dot(Hi,tt)
                        x1=int(tmp[0]/tmp[2])
                        y1=int(tmp[1]/tmp[2])

                        transformed[i,j,:] = im[y1,x1,:]

    plt.imsave(output_dir + "/transformed.png",transformed)


def main():
    user_input = sys.argv

    input_dir  = user_input[3]
    output_dir = user_input[2]

    #THESE HAVE TO COME FROM ARUCO MARKERS!
    coords_BEV = [(185, 20), (155, 295), (92, 51), (78, 215)]   #COORDINATES FROM THE BIRDS EYE VIEW
    coords_OG  = [(160, 25), (160, 250), (350, 25), (350, 250)]

    get_plane(path_to_image='./test_files/og.png', coords_OG=coords_OG, coords_BEV=coords_BEV, output_dir=output_dir)

    print('files in input directory: ', os.listdir(input_dir))

if __name__ == '__main__':
    main()
