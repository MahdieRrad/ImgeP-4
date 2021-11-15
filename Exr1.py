import cv2
import cvzone

class Face_detection():
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier('pack\haarcascade_frontalface_default.xml')
        self.eyes_detection = cv2.CascadeClassifier('pack\haarcascade_eye.xml')
        self.lip_detection = cv2.CascadeClassifier('pack\haarcascade_smile.xml')
        self.flag= 0
        
        cam = cv2.VideoCapture(0)
        while True:
            ret , self.frame = cam.read()
            if ret==False:
                break
            cv2.putText(self.frame,'Emoji on my face:   1',(15,20),cv2.FONT_HERSHEY_SIMPLEX,0.5, (200, 255, 100),2,cv2.LINE_4)
            cv2.putText(self.frame,'Emoji on my lips & eyes:   2',(15,50),cv2.FONT_HERSHEY_SIMPLEX,0.5, (200, 255,100),2,cv2.LINE_4)
            cv2.putText(self.frame,'Pixelate my face:   3',(15,80),cv2.FONT_HERSHEY_SIMPLEX,0.5, (200, 255, 100),2,cv2.LINE_4)
            cv2.putText(self.frame,'Flip:   4',(15,110),cv2.FONT_HERSHEY_SIMPLEX,0.5, (200, 255, 100),2,cv2.LINE_4)
            cv2.putText(self.frame,'Paint my:   5',(15,140),cv2.FONT_HERSHEY_SIMPLEX,0.5, (200, 255, 100),2,cv2.LINE_4)
            cv2.putText(self.frame,'Exit:   6',(15,170),cv2.FONT_HERSHEY_SIMPLEX,0.5, (200, 255, 100),2,cv2.LINE_4)

            self.key = cv2.waitKey(1)
            if self.key==ord('1') or self.flag==1:
                self.emoji_face()
            if self.key==ord('2') or self.flag==2:
                self.emoji_eyes()
            if self.key==ord('3') or self.flag==3:
                self.pixelate_face()
            if self.key==ord('4') or self.flag==4:
                self.flip()
            if self.key==ord('5') or self.flag==5:
                self.paint()
            if self.key==ord('6'):
                break
            cv2.imshow('Webcam',self.frame)

    def emoji_face(self):
        self.flag = 1
        emoji = cv2.imread('emoji/up1.png',cv2.IMREAD_UNCHANGED)
        faces = self.face_detector.detectMultiScale(self.frame , 1.3, minNeighbors=10)
        for face in faces:
            x, y, w, h = face
            emoji = cv2.resize(emoji,(w,h))
            self.frame = cvzone.overlayPNG(self.frame,emoji,[x,y])
            return self.frame

    def emoji_eyes(self):
        self.flag = 2
        eye_emoji = cv2.imread('emoji\eyes.png',cv2.IMREAD_UNCHANGED)
        eyes = self.eyes_detection.detectMultiScale(self.frame , 1.3, minNeighbors=45)
        for eye in eyes:
            x, y, w, h = eye
            try:
                self.frame = cvzone.overlayPNG(self.frame,eye_emoji,[x,y])
            except:
                pass
            
        lip_img =  cv2.imread('emoji\smile.png',cv2.IMREAD_UNCHANGED)
        lips = self.lip_detection.detectMultiScale(self.frame , 1.8,minNeighbors=22)
        for lip in lips:
            lx, ly, lw, lh = lip
            try:
                self.frame = cvzone.overlayPNG(self.frame,lip_img,[lx+10,ly-15])
            except:pass
        return self.frame

    def pixelate_face(self):
        self.flag = 3
        faces = self.face_detector.detectMultiScale(self.frame , 1.3, minNeighbors=5)
        for face in faces:
            x, y, w, h = face
            blur = self.frame[y:y+h,x:x+w]
            pixlate = cv2.resize(blur, (20,20), interpolation=cv2.INTER_LINEAR)
            output = cv2.resize(pixlate, (w, h), interpolation=cv2.INTER_NEAREST)
            self.frame[y:y+h,x:x+w] = output
            return self.frame
        
    def flip(self):
        self.flag = 4
        self.frame = cv2.flip(self.frame,1)
        return self.frame

    def paint(self):
        self.flag = 5
        blur = cv2.GaussianBlur(self.frame,(21,21),0)
        self.frame = self.frame / blur
        return self.frame


Face_detection()