import cv2
import os
import dlib

import subprocess
import shutil
from shutil import copyfile
import sys

import numpy as np
import time

from skimage.transform import resize
from skimage.color import rgb2gray
from numpy import reshape


from matplotlib import pyplot as plt

import re
def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def progressBar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()



def TRIALextractFullBodyFromVideo(path, savePath):
    i = 0
    j = 100
    videos = sorted_nicely(os.listdir(path + "/"))
    if '.DS_Store' in videos:
        videos.remove('.DS_Store')

    for video in videos:
        start_time = time.time()


        videoPath = path + "/" + video
        print("- Processing Video:", videoPath + " ...")
        dataX = []

        copyTarget = "tmp/current_video.mp4"
        if not os.path.exists("tmp/"):
            os.makedirs("tmp/")

        print("--- Copying file:", videoPath + " ...")
        copyfile(videoPath, copyTarget)
        cap = cv2.VideoCapture(copyTarget)

        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        numberOfImages = 0
        check = True
        flag = True
        imageNumber = 0
        print("- Extracting full body:", str(totalFrames) + " Frames ...")

        savePathActorImg = savePath + "/" + video[:-4] + "/Actor_img/"
        savePathSubjectImg = savePath + "/" + video[:-4] + "/Subject_img/"

        savePathActorImg = "prove/"
        savePathSubjectImg = savePathActorImg
        i+=1
        j+=1

        # Fixing bounding boxes coordinates if needed
        if video[:-4] == "Subject_2_Story_8":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=-20, tag="actor")
        elif video[:-4] == "Subject_4_Story_4":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=-20, tag="actor")
        elif video[:-4] == "Subject_4_Story_5":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=80, tag="actor")
        else:
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(tag="actor")


        if video[:-4] == "Subject_1_Story_5":
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(x_shift=-50, tag="subject")
        elif video[:-4] == "Subject_2_Story8":
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(x_shift=-20, tag="subject")
        else:
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(tag="subject")

        if not os.path.exists(savePathActorImg):
            os.makedirs(savePathActorImg)
            #os.makedirs(savePathSubjectImg)

        while (check):
            check, img = cap.read()
            if img is not None:

                #Extract actor face
                imageActor = img[y_act_1:y_act_2, x_act_1:x_act_2]
                cv2.imwrite(savePathActorImg + "/%s_%d_%s.png" % ("A",imageNumber,video[:-4]), imageActor)

                # Extract Subject Face
                imageSubject = img[y_sub_1:y_sub_2, x_sub_1:x_sub_2]
                cv2.imwrite(savePathSubjectImg + "/%s_%d_%s.png" % ("S",imageNumber,video[:-4]), imageSubject)

                imageNumber = imageNumber + 1
                progressBar(imageNumber, totalFrames)

            if imageNumber == 1:
                break
        print('\nRunning time: %f seconds\n' %(time.time() - start_time))


def extractFullBodyFromVideo(path, savePath):

    videos = sorted_nicely(os.listdir(path + "/"))
    if '.DS_Store' in videos:
        videos.remove('.DS_Store')

    for video in videos:
        start_time = time.time()

        videoPath = path + "/" + video
        print("- Processing Video:", videoPath + " ...")
        dataX = []

        copyTarget = "tmp/current_video.mp4"
        if not os.path.exists("tmp/"):
            os.makedirs("tmp/")

        print("--- Copying file:", videoPath + " ...")
        copyfile(videoPath, copyTarget)
        cap = cv2.VideoCapture(copyTarget)

        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        numberOfImages = 0
        check = True
        flag = True
        imageNumber = 0
        print("- Extracting full body:", str(totalFrames) + " Frames ...")

        savePathActorImg = savePath + "/" + video[:-4] + "/Actor_img/"
        savePathSubjectImg = savePath + "/" + video[:-4] + "/Subject_img/"


        # Fixing bounding boxes coordinates if needed
        if video[:-4] == "Subject_2_Story_8":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=-20, tag="actor")
        elif video[:-4] == "Subject_4_Story_4":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=-20, tag="actor")
        elif video[:-4] == "Subject_4_Story_5":
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(x_shift=80, tag="actor")
        else:
            x_act_1, y_act_1, x_act_2, y_act_2 = define_frames(tag="actor")


        if video[:-4] == "Subject_1_Story_5":
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(x_shift=-50, tag="subject")
        elif video[:-4] == "Subject_2_Story8":
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(x_shift=-20, tag="subject")
        else:
            x_sub_1, y_sub_1, x_sub_2, y_sub_2 = define_frames(tag="subject")

        if not os.path.exists(savePathActorImg):
            os.makedirs(savePathActorImg)
            os.makedirs(savePathSubjectImg)

            while (check):
                check, img = cap.read()
                if img is not None:

                    #Extract actor face
                    #try:
                    imageActor = img[y_act_1:y_act_2, x_act_1:x_act_2]
                    imageActor = rgb2gray(resize(imageActor,(size,size))).reshape(size,size,1)
                    cv2.imwrite(savePathActorImg + "/%d.png" % imageNumber, imageActor*255)
                    #except:
                    #print("------error1!")

                    # Extract Subject Face
                    #try:
                    imageSubject = img[y_sub_1:y_sub_2, x_sub_1:x_sub_2]
                    imageSubject = rgb2gray(resize(imageSubject,(size,size))).reshape(size,size,1)
                    cv2.imwrite(savePathSubjectImg + "/%d.png" % imageNumber, imageSubject*255)

                    #except:
                    #print("------error2!")

                    imageNumber = imageNumber + 1
                    progressBar(imageNumber, totalFrames)

            print('\nRunning time: %f seconds\n' %(time.time() - start_time))

def define_frames(tag, size = 620, x_shift = 0, y_shift = 0):

    if (tag=="actor"):
        start_x = 290 + x_shift
    elif (tag=="subject"):
        start_x = 1460 + x_shift
    else:
        raise Exception("Specify a tag!")

    start_y   = 720 - size + y_shift

    end_x   =   start_x + size
    end_y   =   start_y + size

    return start_x, start_y, end_x, end_y


if __name__ == "__main__":

    size = 128

    #Path where the videos are
    path = "../omg_dataset/OMG_Empathy2019/All/Videos"

    #Path where the faces will be saved
    savePath ="full_body_extracted"

    # first, run TRIALextractFullBodyFromVideo(..), to visualize the default bounding box for each video at frame 0.
    # In the code above, edit the bounding box for each video, if needed.
    # When the bounding boxes are properly fixed, you can run extractFullBodyFromVideo(..)

    TRIALextractFullBodyFromVideo(path, savePath)
    #extractFullBodyFromVideo(path, savePath)
