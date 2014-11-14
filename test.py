import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import cv2.cv as cv
import os
import sys


IMAGE_SIZE = (1632, 1224)
V_POINT = (771, 849)
RATIO = 1
INIT_CAM = [0, 0, -1]


def read_file():
    points = [[float(x) for x in data.strip().split()[3:]] for data in open('model.txt')]
    mask = [[float(x) for x in data.strip().split()[1:3]] for data in open('model.txt')]
    return points, mask
def read_file_camera():
    points = [ [float(x) for x in data.strip().split()] for data in open('cameraposition.txt')]
    mask = [[float(x) for x in data.strip().split()[1:3]] for data in open('cameraposition.txt')]
    return points, mask

def chunks(l, n):
    if n < 1:
        n = 1
    return [l[i:i + n] for i in range(0, len(l), n)]


data = read_file()[0]
mask = read_file()[1]

data_line = chunks(data, 4)

# print data_line

# Camera Constants for Part 2

F = 1
B_U = 1
B_V = 1
U_0 = 0
V_0 = 0


# 1.1 Define the Shape (Credits: Tay Yang Shun for pts_set_2())


def pts_set_2():
    def create_intermediate_points(pt1, pt2, granularity):
        new_pts = []
        vector = np.array([(x[0] - x[1]) for x in zip(pt1, pt2)])
        return [(np.array(pt2) + (vector * (float(i) / granularity))) for i in range(1, granularity)]

    pts = []
    granularity = 20

    # Create cube wireframe
    pts.extend(data)

    # print data_line

    # for l in data_line:
    # 	pts.extend(create_intermediate_points(l[0], l[1], granularity))
    # 	pts.extend(create_intermediate_points(l[1], l[2], granularity))
    # 	pts.extend(create_intermediate_points(l[2], l[3], granularity))
    # 	pts.extend(create_intermediate_points(l[3], l[0], granularity))

    return np.array(pts)


pts = pts_set_2()  # include more points to produce wireframe


def normalize(vector):
    mag = np.linalg.norm(vector)
    unit = [x / mag for x in vector]
    return unit


def getCamCoor(pos):
    origin = np.zeros([3, 1])
    camZ = normalize((- pos + origin).T)
    camX = normalize(np.matrix([-camZ[0].item(2), 0, camZ[0].item(0)]))
    camY = normalize(np.cross(camX, camZ))

    return camX[0][0], camY[0][0], camZ[0][0]


# def persView(quatmat, ax):
def persView(quatmat, ax):
    tf = quatmat
    # print "\ncamera position = \n"
    # print tf
    camCoor = getCamCoor(quatmat)
    # print camCoor
    # print "/"
    # print quatmat
    i_f = np.matrix(camCoor[0]).T
    j_f = np.matrix(camCoor[1]).T
    k_f = np.matrix(camCoor[2]).T
    result = []
    for sp in pts:
        sp = np.matrix(sp).T
        sp_tf_T = (sp - tf).T
        # print (F * sp_tf_T * i_f)
        if sp_tf_T * k_f==0:
            print "00000"
        u = ((F * sp_tf_T * i_f) * B_U / (sp_tf_T * k_f)) + U_0
        v = ((F * sp_tf_T * j_f) * B_V / (sp_tf_T * k_f)) + V_0
        result.append((640 - u[0, 0] * 500, 350 - v[0, 0] * 500))
        if (ax is not None):
            ax.plot(u[0, 0], v[0, 0], 'b.')
    # 	#print sp

    result = chunks(result, 4)
    if (ax is not None):
        return ax
    else:
        return result

    # ax.axis([-1, 1, -1, 1])


def persPlot():
    # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    axis = [ax1, ax2, ax3, ax4]
    for i in xrange(0, 4):
        persView(np.matrix([0, i * 10 + 1, -10]).T, axis[i])

    plt.savefig('frame_' + str(i) + '.png')
    plt.clf()


persPlot()

# build 3d

def isZero(array):
    return array[0] == 0 and array[1] == 0 and array[2] == 0


def append(first, second):
    for x in xrange(0, 400):
        for y in xrange(0, 400):
            if isZero(first[x][y]):
                first[x][y] = second[x][y]
    return first


def append2(first, second):
    grey = cv2.cvtColor(first, cv.CV_RGB2GRAY)
    ret, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.add(second, first, first, mask=mask_inv)
    return result


dst = []

maskPoints = [(x[0], x[1]) for x in mask]
planes = chunks(maskPoints, 4)
img = cv2.imread('project.png')
camera,maskcam=read_file_camera()
f=1

for pos in camera:

    canvas = persView(np.matrix([pos[0], pos[1], pos[2]]).T, None)

    ks = [0,1,4,5,2,3,93,94,34,35,36,37,38,39,21,22,23,16,18,17,19,95,20,24,30,31,32,25,27,26,28,29,33,10,11,9,14,15,12,13,58,59,60,61,62,63,64,51,52,53,54,55,56,57,40,41,42,43,44,45,46,47,48,49,50,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92]
    curr = 0
    for k in ks:
        total = len(ks)
        c = np.float32([canvas[k]])
        # print "c = "
        # print c
        p = np.float32([planes[k]])
        # print "p = "
        # print p
        mask = np.zeros(img.shape, dtype=np.uint8)
        roi_corners = np.array(p, dtype=np.int32)
        white = (255, 255, 255)
        cv2.fillPoly(mask, roi_corners, white)

        source = cv2.bitwise_and(img, mask)

        M = cv2.getPerspectiveTransform(p, c)
        if (k == 0):
            dst = cv2.warpPerspective(source, M, (1280, 800))
        else:
            dst = append2(dst, cv2.warpPerspective(source, M, (1280, 800)))
        # dst = cv2.warpPerspective(source,M,(400,400))
        curr = curr + 1
        sys.stdout.write("\rprocessing frame " + str(f) + ": " + ("%.2f" % (100.0 * curr / total)) + '%')
        sys.stdout.flush()

    filename = 'results/frame_' + str(f) + '.jpg'
    cv2.imwrite(filename, dst)
    f+=1


# cv2.imshow("image", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print f
	
