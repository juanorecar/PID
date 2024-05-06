import cv2
import os
import numpy as np

path = 'images'

# Importar las imagenes
images = []
classNames = []
myList = os.listdir(path)

orb = cv2.ORB_create(nfeatures=5000)

for cl in myList:
    imgCurr = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCurr)
    classNames.append(os.path.splitext(cl)[0])

def findDes(images):
    descList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        descList.append(des)
    return descList

descList = findDes(images)

def findID(imgTest, descList, thresh=0.5):
    kp2, des2 = orb.detectAndCompute(imgTest, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in descList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass

    if len(matchList) != 0:
        if max(matchList) > thresh:
            finalVal = matchList.index(max(matchList))

    return finalVal

def detect_cards_in_image(image, descList):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img_gray.shape

    # Umbralización adaptativa
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    cards = []

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Ajusta este valor según el tamaño de las cartas en tu imagen
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                card_roi = image[y:y+h, x:x+w]
                card_id = findID(cv2.cvtColor(card_roi, cv2.COLOR_BGR2GRAY), descList)
                if card_id != -1:
                    cards.append((classNames[card_id], approx))

    return cards

if __name__ == "__main__":
    imageTest = cv2.imread('images_test/cartas.jpg')
    detected_cards = detect_cards_in_image(imageTest, descList)

    for card_name, contour in detected_cards:
        cv2.drawContours(imageTest, [contour], -1, (0, 255, 0), 2)
        cv2.putText(imageTest, card_name, (contour[0][0][0], contour[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Detected Cards', imageTest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
