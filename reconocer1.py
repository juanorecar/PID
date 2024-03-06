import cv2
path = r'C:\Users\Juan\Desktop\manocartas.jpg'
img = cv2.imread(path)


blur = cv2.blur(img, (6,6))
#Binarizar la imagen
imgCanny = cv2.Canny(blur, 10, 150)
#Encontrar contornos
contornos, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#Dibujar contornos
cv2.drawContours(img, contornos, -1, (0,255,0), 10)
#Mostra imagen
cv2.imshow("Blur", imgCanny )
cv2.imshow("Contornos de las cartas", img)
cv2.waitKey(0)
cv2.destroyAllWindows()