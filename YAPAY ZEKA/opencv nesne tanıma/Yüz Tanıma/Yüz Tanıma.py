## @Güneş Nur ÇETİN



import cv2

# Yüz sınıflandırıcıyı yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Kamera ile video akışını başlat
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir frame oku
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Gri tonlamalı frame oluştur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit et
    face_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    
    # Tespit edilen yüzlerin etrafına yeşil dikdörtgen çiz
    for (x, y, w, h) in face_rect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # (0, 255, 0) yeşil renk
    
    # Frame'i ekranda göster
    cv2.imshow("Face Detection", frame)
    
    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
