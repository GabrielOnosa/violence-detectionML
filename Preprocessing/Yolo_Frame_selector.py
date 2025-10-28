import torch
import cv2
import numpy as np


def proximity_score(keypointsA, keypointsB):
    keypoint_ids = [ 7,8,9,10 ] #- indicii corespunzatori wrist si elbow
    jointsA = keypointsA[keypoint_ids, :2]
    jointsB = keypointsB[keypoint_ids, :2]

    # calculez distanta intre perechi de pcte cheie wrist - wrist elbow - elbow
    # incerc sa gasesc minimu si vad eu cum fac scorul - aici am ales efectiv distanta minima

    dists = []
    for id in [0,1,2,3]:
        dists.append(torch.linalg.norm(jointsA[id] - jointsB[id]).to('cpu'))
    dists = torch.from_numpy(np.array(dists))
    return dists.min().item()

def box_overlap_score(boxA, boxB):
    #fie A,B punctele ce reprezinta coltul stanga sus respectiv dreapta jos
    #pentru intersectia dintre cele 2 bounding box-uri

    boxA = boxA.reshape(4)
    boxB = boxB.reshape(4)

    xA = torch.max(boxA[0], boxB[0])
    yA = torch.max(boxA[1], boxB[1])
    xB = torch.min(boxA[2], boxB[2])
    yB = torch.min(boxA[3], boxB[3])

    intersection_area = torch.abs((xB - xA) * (yB - yA))
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = (intersection_area / float(boxA_area + boxB_area - intersection_area + 1e-6)).item()

    return iou

def yolo_frame_selector(video, yolo_pose):
    videoCap = cv2.VideoCapture(video)
    distante = []
    while True:
        ret, frame = videoCap.read()
        if not ret:
            break
        results = yolo_pose(frame, verbose = False)
        results_boxes = results[0].boxes
        results = results[0].keypoints.data
        if results is None or len(results) < 2:
            continue
        distante_frame = []
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                distanta = proximity_score(results[i], results[j])
                intersectie = box_overlap_score(results_boxes[i].xyxy, results_boxes[j].xyxy)
                distante_frame.append(distanta + intersectie)
        distante.append(min(distante_frame))
    distante = np.array(distante)
    if len(distante) >= 3:
        return np.argsort(distante)[:3]
    else:
        return [38,76,114]


if __name__ == '__main__':
    import pandas as pd
    import os

    for
