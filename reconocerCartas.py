import cv2
import numpy as np

path = r'C:\Users\Juan\Desktop\cartas.png'
path2 = r'C:\Users\Juan\Desktop\manocartas.jpg'

img = cv2.imread(path)
img2 = cv2.imread(path2)


gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
_,th = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

contornos1,hierarchy1 = cv2.findContours(th, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)
contornos2,hierarchy2 = cv2.findContours(th, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img2, contornos1, -1, (0,255,0), 3)
#print ('len(contornos1[2])=',len(contornos1[2]))
#print ('len(contornos2[2])=',len(contornos2[2]))
cv2.imshow('imagen',img2)
cv2.imshow('th',th)
cv2.waitKey(0)
cv2.destroyAllWindows()


grises = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grises2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

bordes = cv2.Canny(grises, 100, 200)
bordes2 = cv2.Canny(grises2, 100, 200)

ctns, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, ctns, -1, (0,0,255), 2)

ctns2, _ = cv2.findContours(bordes2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img2, ctns, -1, (0,0,255), 2)

#print('Número de contornos encontrados: ', len(ctns))


cv2.imshow("Bordes",bordes)
cv2.imshow("Bordes2",bordes2)
cv2.imshow("Imagen",img)
cv2.imshow("Imagen2",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()