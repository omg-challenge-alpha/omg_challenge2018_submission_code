import cv2
import os
import dlib

import subprocess
import shutil
from shutil import copyfile
import sys

import numpy as np
import time

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


def extractFramesFromVideo(path,savePath, faceDetectorPrecision):
    videos = sorted_nicely(os.listdir(path + "/"))
    if '.DS_Store' in videos:
        videos.remove('.DS_Store')


    for video in videos:
        video = video[:-4]
        start_time = time.time()

        videoPath = path + "/" + video + ".mp4"
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
        lastImageWithFaceDetected = 0
        print("- Extracting Faces:", str(totalFrames) + " Frames ...")

        savePathActorFace = savePath + "/" + video + "/Actor_face/"
        savePathSubjectFace = savePath + "/" + video + "/Subject_face/"

        savePathActorFaceXY = savePath + "/" + video + "/Actor_face_coordinates/"
        savePathActorFaceLandmarks = savePath + "/" + video + "/Actor_face_landmarks/"

        savePathSubjectFaceXY = savePath + "/" + video + "/Subject_face_coordinates/"
        savePathSubjectFaceLandmarks = savePath + "/" + video + "/Subject_face_landmarks/"


        faceActor = np.zeros((totalFrames, 4), dtype=int)
        landmarksActor = np.zeros((totalFrames, 68*2), dtype=int)

        faceSubject = np.zeros((totalFrames, 4), dtype=int)
        landmarksSubject = np.zeros((totalFrames, 68*2), dtype=int)

        if not os.path.exists(savePathActorFace):
            os.makedirs(savePathActorFace)
            os.makedirs(savePathActorFaceXY)
            os.makedirs(savePathActorFaceLandmarks)

            os.makedirs(savePathSubjectFace)
            os.makedirs(savePathSubjectFaceXY)
            os.makedirs(savePathSubjectFaceLandmarks)

        while (check):
            check, img = cap.read()
            if img is not None:


                #Extract actor face
                imageActor = img[0:720, 0:1280]

                if lastImageWithFaceDetected == 0 or lastImageWithFaceDetected > faceDetectorPrecision:
                    dets = detector(imageActor, 1)
                    lastImageWithFaceDetected = 0

                    if not len(dets) == 0:
                        oldDetsActor = dets
                else:
                    dets = oldDetsActor

                try:
                    if not len(dets) == 0:
                        for i, d in enumerate(dets):
                            croped = imageActor[d.top():d.bottom(), d.left():d.right()]
                            cv2.imwrite(savePathActorFace + "/%d.png" % imageNumber, croped)

                            faceActor[imageNumber] = rects_to_np(d)

                            landmarks = dlib_determine_landmarks(d, imageActor)
                            landmarksActor[imageNumber] = landmarks.flatten()

                    else:
                        if not imageNumber == 0:
                            faceActor[imageNumber] = faceActor[imageNumber-1]
                            landmarksActor[imageNumber] = landmarksActor[imageNumber-1]
                        cv2.imwrite(savePathActorFace + "/%d.png" % imageNumber, imageActor)


                except:
                    print("------error1!")

                # Extract Subject Face
                imageSubject = img[0:720, 1280:2560]

                if lastImageWithFaceDetected == 0 or lastImageWithFaceDetected > faceDetectorPrecision:
                    dets = detector(imageSubject, 1)
                    lastImageWithFaceDetected = 0

                    if not len(dets) == 0:
                        oldDetsSubject = dets
                else:
                    dets = oldDetsSubject

                try:
                    if not len(dets) == 0:
                        for i, d in enumerate(dets):
                            croped = imageSubject[d.top():d.bottom(), d.left():d.right()]
                            cv2.imwrite(savePathSubjectFace + "/%d.png" % imageNumber, croped)

                            faceSubject[imageNumber] = rects_to_np(d) + np.array([1280,0,1280,0])

                            landmarks = dlib_determine_landmarks(d, imageSubject) + np.array([1280,0])
                            landmarksSubject[imageNumber] = landmarks.flatten()

                            # Just for testing..
                            if imageNumber == 0 and False:
                                plt.figure()
                                #plt.imshow(cv2.cvtColor(draw_pic, cv2.COLOR_BGR2RGB))
                                cv2.rectangle(img, (faceSubject[imageNumber][0],faceSubject[imageNumber][1]),
                                (faceSubject[imageNumber][2],faceSubject[imageNumber][3]), (255,0,0), 2)
                                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                                plt.title('dlib landmark points')
                                plt.show()
                            if imageNumber == 0 and False:
                                # show the output image with the face detections + facial landmarks
                                draw_pic = img.copy()
                                # loop over the (x, y)-coordinates for the facial landmarks
                                # and draw them on the image
                                for (x, y) in landmarks:
                                    cv2.circle(draw_pic, (x, y), 1, (0, 0, 255), -1)
                                plt.figure()
                                plt.imshow(cv2.cvtColor(draw_pic, cv2.COLOR_BGR2RGB))
                                plt.title('dlib landmark points')
                                plt.show()

                    else:
                        if not imageNumber == 0:
                            faceSubject[imageNumber] = faceSubject[imageNumber-1]
                            landmarksSubject[imageNumber] = landmarksSubject[imageNumber-1]

                        cv2.imwrite(savePathSubjectFace + "/%d.png" % imageNumber, imageSubject)

                except:
                    print("------error2!")

                imageNumber = imageNumber + 1
                lastImageWithFaceDetected = lastImageWithFaceDetected + 1
                progressBar(imageNumber, totalFrames)

        np.savetxt(savePathActorFaceXY + "faceActor.csv", faceActor.astype(int), fmt='%i', delimiter=",")
        np.savetxt(savePathActorFaceLandmarks + "landmarksActor.csv", landmarksActor.astype(int), fmt='%i', delimiter=",")

        np.savetxt(savePathSubjectFaceXY + "faceSubject.csv", faceSubject.astype(int), fmt='%i', delimiter=",")
        np.savetxt(savePathSubjectFaceLandmarks + "landmarksSubject.csv", landmarksSubject.astype(int), fmt='%i', delimiter=",")

        print('\nRunning time: %f seconds\n' %(time.time() - start_time))

def dlib_determine_landmarks(rect, img):
    # returns the landmarks of each face
    # landmarks has shape [#landmarks=68, #coordinates=2]

    # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    landmarks = np.zeros([68,2],dtype=int)

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array

    shape = predictor(img, rect)

    landmarks = shape_to_np(shape)

    if 0:
        # show the output image with the face detections + facial landmarks
        draw_pic = img.copy()
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in landmarks:
                cv2.circle(draw_pic, (x, y), 1, (0, 0, 255), -1)
        plt.figure()
        plt.imshow(cv2.cvtColor(draw_pic, cv2.COLOR_BGR2RGB))
        plt.title('dlib landmark points')
        plt.show()

    #print(landmarks.shape)
    return landmarks

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


def rects_to_np(rect):

    # take a bounding predicted by dlib and convert it
    # to the format (x1, y1, x2, y2), where (x1,y1) is a vertex of the
    # rectangle and (x2,y2) is the vertex opposite to (x1,y1))

    faceRects = np.zeros([1,4],dtype=int)

    x1 = int(rect.left())
    y1 = int(rect.top())
    x2 = int(rect.right())
    y2 = int(rect.bottom())
    faceRects = np.array([[x1,y1,x2,y2]])
    #print(faceRects.shape)
    return faceRects




if __name__ == "__main__":


    #Path where the videos are
    path = "../omg_dataset/OMG_Empathy2019_testSet/Videos"

    #Path where the faces will be saved
    savePath ="faces_extracted/"

    # If 1, the face detector will act upon each of the frames. If 1000, the face detector update its position every 1000 frames.
    # Actually: if 9, the face detector update its position every 10 frames.
    faceDetectorPrecision = 9

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    extractFramesFromVideo(path, savePath, faceDetectorPrecision)
