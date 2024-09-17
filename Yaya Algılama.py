##  @Güneş Nur ÇETİN


import cv2

# HOG tanımlayıcısı oluştur
hog = cv2.HOGDescriptor()
# Önceden eğitilmiş SVM modelini HOG tanımlayıcısına ata
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Gerçek zamanlı kamera akışını başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı belirtir

# Kamera kaynağının başarıyla açıldığından emin ol
if not cap.isOpened():
    print("Kamera açılmadı. Kamera bağlantısını kontrol edin.")
    exit()

# Kamera akışını işleme döngüsü
while cap.isOpened():
    # Kamera'dan bir kare oku
    ret, frame = cap.read()
    if not ret:
        # Eğer kare okunamazsa döngüden çık
        print("Kare okunamadı.")
        break

    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gri tonlamalı görüntüde yayaları tespit et
    (rects, weights) = hog.detectMultiScale(gray, padding=(4, 4), scale=1.1)
    
    # Tespit edilen bölgeleri çiz
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # İşlenmiş görüntüyü ekranda göster
    cv2.imshow("Yaya Tespiti", frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kaynağını serbest bırak ve tüm pencereleri kapat
cap.release()
cv2.destroyAllWindows()
