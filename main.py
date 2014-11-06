import cv2
import numpy as np
from Tkinter import *
from random import randint
from math import floor
z = 0
# set of all 3D points
M = np.zeros((1000, 6))
N = 0

def write_file():
    file = open("dataset.txt", "w")
    for i in range(0, N):
        file.write('{0} {1} {2} {3} {4} {5}'.format(*M[i,:]))        
        file.write("\n")
    file.close()
    
def set_depth():
    master = Tk()
    e = Entry(master)
    f = Entry(master)
    g = Entry(master)
    e.pack()
    f.pack()
    g.pack()
    e.focus_set()
    def close_window():
        global z
        global x
        global y
        x = e.get()
        y = f.get()
        z = g.get()
        master.destroy()
    b = Button(master, text = "OK", width = 10, command = close_window)
    b.pack()
    mainloop()

def gen_color():
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    return (r, g, b)

# mouse callback function
def draw_circle(event,u,v,flags,param):
    global M
    global N
    global C
    if event == cv2.EVENT_LBUTTONDOWN:
        if N % 4 == 0:
            C = gen_color()
        set_depth()
        M[N,:] = [int(floor(N)), u, v, x, y, z]
        cv2.circle(img,(u,v),4,C,2)
        N = N + 1
        print x, y, z
        
# Load an color image in grayscale
img = cv2.imread('project.png',cv2.IMREAD_COLOR)
height, width, depth = img.shape
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 600*width/height, 600);
cv2.setMouseCallback('image',draw_circle)

height =  img.shape
print height

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        M[:,0] = M[:,0]
        M[:,1] = M[:,1]
        write_file()
        break

cv2.destroyAllWindows()

