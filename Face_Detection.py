#MADE BY SHRAVAN


import cv2


face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cv2.namedWindow("meow", cv2.WINDOW_NORMAL)
cv2.resizeWindow("meow", 700, 600)


#webcam

webcam = cv2.VideoCapture(0)
a=0
while True:
    success_frame,frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coord = face_data.detectMultiScale(gray,1.3,5)

    for i in range(len(coord)):
        [x, y, w, h] = coord[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        #cropped = frame[ y:y + h, x:x + w]


    cv2.imshow("meow", frame)

    print("TOTAL FACES DETECTED:", len(coord))
    key = cv2.waitKey(1)
    if key == 27:  #ESC KEY TO EXIT
        break
cv2.destroyAllWindows()



#image
'''
img = cv2.imread("img.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
coord = face_data.detectMultiScale(gray,1.3,5)


for i in range(len(coord)):
    [x,y,w,h] = coord[i]
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 5)


print("TOTAL FACES DETECTED:",len(coord))
cv2.imshow("cutu",img)
cv2.waitKey()
cv2.destroyAllWindows()

'''


