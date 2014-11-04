import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import cv2.cv as cv

# Camera Constants for Part 2

F = 1
B_U = 1
B_V = 1
U_0 = 0
V_0 = 0



def process(imagePoints, modelPoints):

	maskPoints = [[(x[0], x[1]) for x in imagePoints]]

	source = cv2.imread('test.jpg')
	pts1 = []
	pts1.extend(imagePoints)
	pts1 = np.float32(pts1)

	mask = np.zeros(source.shape, dtype=np.uint8)
	roi_corners = np.array(maskPoints, dtype=np.int32)
	white = (255, 255, 255)
	cv2.fillPoly(mask, roi_corners, white)

	source = cv2.bitwise_and(source, mask)

	def pts_set_2():

	  pts = []
	 
	  # Create cube wireframe
	  pts.extend(modelPoints)

	  return np.array(pts)
	 
	pts = pts_set_2() # include more points to produce wireframe


	# 1.2 Define the Camera Translation

	def quatmult(q1, q2):
		out = [0, 0, 0, 0]
		out[0] = q1[0]*q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
		out[1] = q1[0]*q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
		out[2] = q1[0]*q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
		out[3] = q1[0]*q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
		return out

	def normalize(vector):
		mag = np.linalg.norm(vector)
		unit = [x / mag for x in vector]
		return unit

	def createRotQuat(degree, axis):
		# normalize
		unit = normalize(axis)
		r = math.radians(degree)
		q = [0, 0, 0, 0]
		q[0] = math.cos(r/2)
		q[1] = math.sin(r/2) * unit[0]
		q[2] = math.sin(r/2) * unit[1]
		q[3] = math.sin(r/2) * unit[2]
		return q

	def conjQuat(q):
		q_c = [q[0], -q[1], -q[2], -q[3]]
		return q_c

	def calcCamPos(init, degree):
		p = init
		q = createRotQuat(degree, [0, 1, 0])
		q_c = conjQuat(q)
		p_p = quatmult(quatmult(q, p), conjQuat(q))
		return p_p


	# 1.3 Define the Camera Orientation

	initCamPos = np.matrix([0, 0, -5]).T

	def quat2rot(q):
		rot = np.zeros([3, 3])
		q0 = q[0]
		q1 = q[1]
		q2 = q[2]
		q3 = q[3]
		q0_2 = math.pow(q0, 2)
		q1_2 = math.pow(q1, 2)
		q2_2 = math.pow(q2, 2)
		q3_2 = math.pow(q3, 2)
		rot[0, :] = [q0_2 + q1_2 - q2_2 - q3_2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)]
		rot[1, :] = [2 * (q1 * q2 + q0 * q3), q0_2 + q2_2 - q1_2 - q3_2, 2 * (q2 * q3 - q0 * q1)]
		rot[2, :] = [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0_2 + q3_2 - q1_2 - q2_2]

		return np.matrix(rot)

	# Camera Orientation

	quatmat_4 = (quat2rot(createRotQuat(-90, [0, 1, 0])) * initCamPos)
	quatmat_3 = (quat2rot(createRotQuat(-60, [0, 1, 0])) * initCamPos)
	quatmat_2 = (quat2rot(createRotQuat(-30, [0, 1, 0])) * initCamPos)
	quatmat_1 = (quat2rot(createRotQuat(-0, [0, 1, 0])) * initCamPos)

	# 2 Projecting 3D shape points onto camera image planes

	# get camera coordinate system based on its position

	def getCamCoor(pos):
		origin = np.zeros([3, 1])
		camZ = normalize((- pos + origin).T)
		camY = normalize(np.matrix([0, 1, 0]))
		camX = normalize(np.cross(camY, camZ))
		# print camX[0][0], camY[0][0], camZ[0][0]
		return camX[0][0], camY[0][0], camZ[0][0]

	# 2.1 Perspective Model

	def persView(quatmat):
		tf = quatmat
		camCoor = getCamCoor(quatmat)
		i_f = np.matrix(camCoor[0]).T
		j_f = np.matrix(camCoor[1]).T
		k_f = np.matrix(camCoor[2]).T
		result = []
		for sp in pts:
			sp = np.matrix(sp).T 
			sp_tf_T = (sp - tf).T
			u = ((F * sp_tf_T * i_f) * B_U / (sp_tf_T * k_f)) + U_0
			v = ((F * sp_tf_T * j_f) * B_V / (sp_tf_T * k_f)) + V_0
			result.extend([[200 - u[0, 0] * 100, 200 - v[0, 0] * 100]])
		result = np.float32(result)
		# print pts1, result
		M = cv2.getPerspectiveTransform(pts1,result)

		dst = cv2.warpPerspective(source,M,(300,300))

		return dst
		# plt.subplot(121)
		# plt.imshow(source)
		# plt.title('Input')
		# plt.subplot(122)
		# plt.imshow(dst)
		# plt.title('Output')
		# plt.show()

	return persView(quatmat_3)

	# def persPlot():
	# 	persView(quatmat_1)
	# 	persView(quatmat_2)
	# 	persView(quatmat_3)
	# 	persView(quatmat_4)

	# persPlot()


first = process([[300, 300], [100, 300], [100, 100], [300, 100]], [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]])
second = process([[300, 300], [100, 300], [100, 100], [300, 100]], [[-1, -1, 1], [1, -1, 1], [1, -1, -1], [-1, -1, -1]])
third = process([[300, 300], [100, 300], [100, 100], [300, 100]], [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
fourth = process([[300, 300], [100, 300], [100, 100], [300, 100]], [[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]])

def isZero(array):
	return array[0] == 0 and array[1] == 0 and array[2] == 0

def append(first, second):

	for x in xrange(0,300):
		for y in xrange(0,300):
			if isZero(first[x][y]):
				first[x][y] = second[x][y]
	return first

first = append(append(append(first, second), third), fourth)


# final = cv2.add(first, second)



plt.imshow(first)
plt.show()
