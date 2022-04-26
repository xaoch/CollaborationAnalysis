# This is a sample Python script.
import cv2
import numpy as np
import time

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# build the mapping
def buildMap(Ws, Hs, Wd, Hd, R1, R2, Cx, Cy, phi):
    map_x = np.zeros((Hd, Wd), np.float32)
    map_y = np.zeros((Hd, Wd), np.float32)
    for y in range(0, int(Hd - 1)):
        for x in range(0, int(Wd - 1)):
            r = (float(y) / float(Hd)) * (R2 - R1) + R1
            theta = ((float(x) / float(Wd)) * 2.0 * np.pi)+phi
            xS = Cx + r * np.sin(theta)
            yS = Cy + r * np.cos(theta)
            map_x.itemset((y, x), int(xS))
            map_y.itemset((y, x), int(yS))

    return map_x, map_y

# do the unwarping
def unwarp(img, xmap, ymap):
    output = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR)
    #result = Image(output, cv2image=True)
    return output #result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image =cv2.imread("images/0009.png")
    Ws = image.shape[1]
    Hs = image.shape[0]
    print(Ws, Hs)

    Cx = int(Ws/2)
    Cy = int(Hs/2)

    # Inner donut radius
    R1 = 200
    # outer donut radius
    R2 = 540
    # our input and output image siZes
    Wd = int(2.0 * ((R2 + R1) / 2) * np.pi)
    Hd = (R2 - R1)

    # build the pixel map, this could be sped up
    print("BUILDING MAP!")
    print(Ws, Hs, Wd, Hd, R1, R2, Cx, Cy)
    xmap, ymap = buildMap(Ws, Hs, Wd, Hd, R1, R2, Cx, Cy,np.pi/2)
    print("MAP DONE!")
    # do an unwarping and show it to us
    result = unwarp(image, xmap, ymap)
    cv2.imwrite("output.png",result)










