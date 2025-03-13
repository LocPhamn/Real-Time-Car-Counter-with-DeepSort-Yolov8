import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort
from deep_sort_realtime.deepsort_tracker import DeepSort


cap = cv2.VideoCapture("./Video/cars.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8l.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")


#DeepSort Tracking
deepTracker = DeepSort(max_age=5, n_init=3, nms_max_overlap=1.0)


limits = [400, 297, 673, 297]
totalCount = []

while True:
    success, img = cap.read()
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)
    # detections = np.empty((0, 5))
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            class_id = int(box.cls[0])
            currentClass = classNames[class_id]
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                detections.append([[x1,y1,x2-x1,y2-y1],conf,class_id])

        tracks = deepTracker.update_tracks(detections,frame=imgRegion)
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

        for track in tracks:
            track_id = track.track_id

            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1,y1,x2,y2 = map(int,ltrb)
            w, h = x2 - x1, y2 - y1

            label = "{}-{}".format(classNames[class_id],track_id)

            # show tracking id
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)),
                               scale=2, thickness=1, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if totalCount.count(track_id) == 0:
                    totalCount.append(track_id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break
        # boxes = r.boxes
        # for box in boxes:
        #     # Bounding Box
        #     x1, y1, x2, y2 = box.xyxy[0]
        #     x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        #     w, h = x2 - x1, y2 - y1
        #
        #     # Confidence
        #     conf = math.ceil((box.conf[0] * 100)) / 100
        #     # Class Name
        #     cls = int(box.cls[0])
        #     currentClass = classNames[cls]
        #
        #     if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
        #
        #         # object detect, show label and confident
        #         cvzone.putTextRect(img, '{} {:0.2f}'.format(currentClass,conf), (max(0, x1), max(35, y1)),
        #                            scale=1, thickness=1, offset=3)
        #         cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
        #
        #         currentArray = np.array([x1, y1, x2, y2, conf])
        #         detections = np.vstack((detections, currentArray))




    # resultsTracker = tracker.update(detections)
    #
    # for result in resultsTracker:
    #     x1, y1, x2, y2, id = result
    #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #     print(result)
    #     w, h = x2 - x1, y2 - y1
    #
    #     # show tracking id
    #     cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
    #     cvzone.putTextRect(img, ' {}'.format(int(id)), (max(0, x1), max(35, y1)),
    #                        scale=2, thickness=3, offset=10)
    #
    #     cx, cy = x1 + w // 2, y1 + h // 2
    #     cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    #
    #     if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
    #         if totalCount.count(id) == 0:
    #             totalCount.append(id)
    #             cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    # cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # if cv2.waitKey(1) == ord("q"):
    #     break
cap.release()
cv2.destroyWindow()