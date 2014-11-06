import numpy as np
import math
import matplotlib.pyplot as plt


IMAGE_SIZE = (1632, 1224)
V_POINT = (700, 849)
RATIO = 1
INIT_CAM = [0, 0, -1]

def read_file():
	data = [data.strip().split() for data in open('model.txt')]
	print data
	return data

def build_model(data):

	# ratio_x = float(500) / float(1154 - 978)
	# ratio_y = float(500) / float(888 - 874)
	# x_f = ratio_x * float(1154 - V_POINT[0])
	# y_f = ratio_y * float(888 - V_POINT[1])
	# print ratio_x, ratio_y
	# print x_f, y_f

	ratio_x = float(1154 - 978) / float(100)
	ratio_y = float(888 - 874) / float(100)
	print ratio_x, ratio_y


	# def normalize_data(data):
	# 	for i in xrange(0,len(data)):
	# 		z = float(data[i][2])
	# 		x = - (float(data[i][0]) - V_POINT[0])
	# 		y = - (float(data[i][1]) - V_POINT[1])
	# 		# print x, y, z

			
	# 		hx = x_f - z
	# 		hy = y_f - z

	# 		x = x / hx * x_f

	# 		y = y / hy * y_f

	# 		data[i] = [x, y, z]
	# 		x_list = [d[0] for d in data]
	# 		y_list = [d[1] for d in data]
	# 		z_list = [d[2] for d in data]

	# 	return data

	def normalize_data(data):
		result = []
		for i in xrange(0,len(data)):
			# make the (0, 0) to be the vanishing point
			x = float(data[i][0]) - V_POINT[0]
			y = float(data[i][1]) - V_POINT[1]
			z = float(data[i][2])
			# rescale based on z
			if (x < 0):
				x = x - z * ratio_x
			else:
				x = x + z * ratio_x

			if (y < 0):
				y = y - z * ratio_y
			else:
				y = y + z * ratio_y
			
			print (x, y, z)
			result.append([-x, -y, z])
			x_list = [d[0] for d in data]
			z_list = [d[2] for d in data]
		return result


	return normalize_data(data)

data = build_model(read_file())

def chunks(l, n):
    if n < 1:
        n = 1
    return [l[i:i + n] for i in range(0, len(l), n)]



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
		return [(np.array(pt2) + (vector * (float(i)/granularity))) for i in range(1, granularity)]

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

pts = pts_set_2() # include more points to produce wireframe


# Camera Position by Pure Maths Calculation

# camPos = np.zeros([4, 3])
# camPos[0, :] = [0, 0, -5]
# camPos[1, :] = [2.5, 0, -4.33]
# camPos[2, :] = [4.33, 0, -2.5]
# camPos[3, :] = [5, 0, -0]

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
	#print q
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

# initCamQuat = [0, 0, 0, -5]
# rotDegrees = [0, -30, -60, -90]
# camPos = [calcCamPos(initCamQuat, r) for r in rotDegrees] # Camera location in frame 1-4


# 1.3 Define the Camera Orientation

initCamPos = np.matrix(INIT_CAM).T

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

quatmat_4 = (quat2rot(createRotQuat(30, [1, 0, 0])) * initCamPos)
quatmat_3 = (quat2rot(createRotQuat(20, [1, 0, 0])) * initCamPos)
quatmat_2 = (quat2rot(createRotQuat(10, [1, 0, 0])) * initCamPos)
quatmat_1 = (quat2rot(createRotQuat(0, [1, 0, 0])) * initCamPos)

# print "quat1: " 
# print quatmat_2, initCamPos

# 2 Projecting 3D shape points onto camera image planes

# get camera coordinate system based on its position

def getCamCoor(pos):
	origin = np.zeros([3, 1])
	camZ = normalize((- pos + origin).T)
	camY = normalize(np.matrix([0, 1, 0]))
	camX = normalize(np.cross(camY, camZ))
	return camX[0][0], camY[0][0], camZ[0][0]

# 2.1 Perspective Model

def persView(quatmat, ax):
	tf = quatmat
	print "\ncamera position = \n"
	print tf
	camCoor = getCamCoor(quatmat)
	# print "/"
	# print quatmat
	i_f = np.matrix(camCoor[0]).T
	j_f = np.matrix(camCoor[1]).T
	k_f = np.matrix(camCoor[2]).T
	sd = 0
	for sp in pts:
		# print "\npoint = \n"
		# print sp
		sp = np.matrix(sp).T 
		sp_tf_T = (sp - tf).T
		u = ((F * sp_tf_T * i_f) * B_U / (sp_tf_T * k_f)) + U_0
		v = ((F * sp_tf_T * j_f) * B_V / (sp_tf_T * k_f)) + V_0
		ax.plot(u[0, 0], v[0, 0], 'b,')
		print (u[0, 0], v[0, 0])
		#print sp

	ax.axis([(V_POINT[1] - IMAGE_SIZE[1])/RATIO, (IMAGE_SIZE[1] - V_POINT[1])/RATIO, (V_POINT[1] - IMAGE_SIZE[1])/RATIO, (IMAGE_SIZE[1] - V_POINT[1])/RATIO])

def persPlot():
	# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	for i in xrange(0,4):
		j = i * 30
		f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
		q1 = (quat2rot(createRotQuat(j, [1, 0, 0])) * initCamPos)
		q2 = (quat2rot(createRotQuat(j, [0, 1, 0])) * initCamPos)
		q3 = (quat2rot(createRotQuat(j, [0, 0, 1])) * initCamPos)
		q4 = (quat2rot(createRotQuat(0, [1, 0, 0])) * initCamPos)
		persView(q1, ax1)
		persView(q2, ax2)
		persView(q3, ax3)
		persView(q4, ax4)
		plt.savefig('frame_' + str(i) + '.png')
		plt.clf()

persPlot()

# # 2.2 Orthographic Model

# def orthView(quatmat, ax):
# 	tf = quatmat
# 	camCoor = getCamCoor(quatmat)
# 	i_f = np.matrix(camCoor[0]).T
# 	j_f = np.matrix(camCoor[1]).T
# 	for sp in pts:
# 		sp = np.matrix(sp).T 
# 		sp_tf_T = (sp - tf).T
# 		u = ((sp_tf_T * i_f) * B_U )+ U_0
# 		v = ((sp_tf_T * j_f) * B_V ) + V_0
# 		ax.plot(u[0, 0], v[0, 0], 'r.')

# 	ax.axis([-2, 2, -2, 2])

# def orthPlot():
# 	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
# 	orthView(quatmat_1, ax1)
# 	orthView(quatmat_2, ax2)
# 	orthView(quatmat_3, ax3)
# 	orthView(quatmat_4, ax4)
# 	plt.savefig('orth.png')
# 	plt.clf()

# orthPlot()
