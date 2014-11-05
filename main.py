import cv2
import numpy as np
from Tkinter import *
z = 0
# set of all 3D points
M = np.zeros((1000, 3))
N = 0

def write_file():
    file = open("dataset.txt", "w")
    for i in range(0, N):
        file.write('{0} {1} {2}'.format(*M[i,:]))        
        file.write("\n")
    file.close()
    
def set_depth():
    master = Tk()
    e = Entry(master)
    e.pack()
    e.focus_set()
    def close_window():
        global z
        z = int(e.get())
        master.destroy()
    b = Button(master, text = "OK", width = 10, command = close_window)
    b.pack()
    mainloop()

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global M
    global N
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),10,(255,0,0),3)        
        set_depth()
        M[N,:] = [x, y, z]
        N = N + 1
        print x, y, z
        
# Load an color image in grayscale
img = cv2.imread('project.jpeg',cv2.IMREAD_COLOR)
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
        M[:,0] = M[:,0] - width/2
        M[:,1] = M[:,1] - height/2
        write_file()
        break

cv2.destroyAllWindows()
