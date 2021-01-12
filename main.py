import cv2
import numpy as np
import sys
import os


def ocr_cnn(img, resize_width=320, resize_height=320):
    orig = img.copy()
    (H, W) = img.shape[:2]
    rW = W / float(resize_width)
    rH = H / float(resize_height)
    img = cv2.resize(img, (resize_height, resize_width))
    if resize_height % 32 != 0 or resize_width % 32 != 0:
        raise ValueError("resize_width and resize_height must be a multiple of 32 when using cnn approach")
    layer_names = [	"feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    net = cv2.dnn.readNet('./models/frozen_east_text_detection.pb')
    blob = cv2.dnn.blobFromImage(img, 1.0, (resize_width, resize_height),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layer_names)
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(numCols):
            if scoresData[x] < 0.5:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    for (startX, startY, endX, endY) in rects:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 1)

    cv2.imshow("Text detected via CNN", orig)
    cv2.waitKey(0)


def ocr_classical(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(img, (7, 7), 0)
    edged = cv2.Canny(gray, 30, 150)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 10 or h > 10:
            rectangles.append((x, y, x + w, y+h))
    return rectangles


def ocr_line_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    pts = cv2.findNonZero(threshold)
    ret = cv2.minAreaRect(pts)

    (cx, cy), (w, h), ang = ret
    if w < h:
        w, h = h, w
        ang += 90
    m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rotated = cv2.warpAffine(threshold, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    ret_img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))

    hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)

    th = 3
    H, W = img.shape[:2]
    lines = [y for y in range(H-1) if hist[y] <= th < hist[y + 1] or hist[y] > th >= hist[y + 1]]
    return ret_img, W, lines


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Image path must be given as argument")
    for img_path in sys.argv[1:]:
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(sys.argv[1])
        img, W, lines = ocr_line_detection(img)
        rectangles = ocr_classical(img)
        for y in lines:
            cv2.line(img, (0, y), (W, y), (0, 0, 255), 2)
        for (x1, y1, x2, y2) in rectangles:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imwrite(os.path.join('./data/output', 'processed_' + os.path.split(img_path)[1]), img)
