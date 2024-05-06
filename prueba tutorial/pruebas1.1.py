import cv2
import os

path = 'images'

# Importar las imágenes
images = []
classNames = []
myList = os.listdir(path)

orb = cv2.ORB_create(nfeatures=1000)

for cl in myList:
    imgCurr = cv2.imread(os.path.join(path, cl), 0)
    images.append(imgCurr)
    classNames.append(os.path.splitext(cl)[0])

# Función para encontrar descriptores
def find_descriptors(images):
    desc_list = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desc_list.append((kp, des))
    return desc_list

desc_list = find_descriptors(images)

# Función para encontrar el ID de la carta
def find_card_id(img_test, desc_list, thresh=0.7):
    kp2, des2 = orb.detectAndCompute(img_test, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    max_match_count = 0
    max_match_index = -1

    for i, (_, des) in enumerate(desc_list):
        matches = bf.match(des, des2)
        match_count = sum(1 for m in matches if m.distance < 50)
        if match_count > max_match_count:
            max_match_count = match_count
            max_match_index = i

    if max_match_count > thresh * len(desc_list[max_match_index][1]):
        return max_match_index
    else:
        return -1

# Función para detectar cartas en la imagen
def detect_cards_in_image(image, desc_list):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Umbralización adaptativa
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)

    cards = []

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Ajusta este valor según el tamaño de las cartas en tu imagen
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.5 < aspect_ratio < 2.0:  # Ajusta estos valores según la relación de aspecto de las cartas
                    card_roi = image[y:y+h, x:x+w]
                    card_id = find_card_id(cv2.cvtColor(card_roi, cv2.COLOR_BGR2GRAY), desc_list)
                    if card_id != -1:
                        cards.append((classNames[card_id], approx))

    return cards

if __name__ == "__main__":
    imageTest = cv2.imread('images_test/cartas.jpg')
    detected_cards = detect_cards_in_image(imageTest, desc_list)

    for card_name, contour in detected_cards:
        cv2.drawContours(imageTest, [contour], -1, (0, 255, 0), 2)
        cv2.putText(imageTest, card_name, (contour[0][0][0], contour[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Detected Cards', imageTest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
