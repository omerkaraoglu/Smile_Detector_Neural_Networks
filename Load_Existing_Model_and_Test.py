import os
import cv2
import numpy as np
#import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import keras

import matplotlib.pyplot as plt


# Sınıf isimlerini ve giriş yollarını tanımla
directories = ["CustomTestImages/positives", "CustomTestImages/negatives"]

# Resim ve etiket dizilerini oluştur
images = []
labels = []

added_files = set()

for directory in directories:
    for file_name in os.listdir(directory):
        img_path = f"{directory}/{file_name}"

        if img_path in added_files:
            continue

        # Dosyayı siyah-beyaz olarak yükle
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Resmi siyah-beyaz olarak yükle

        # Dosya okunamadıysa bir uyarı ver ve bir sonraki dosyaya geç
        if img is None:
            print(f"Warning: Unable to read file {img_path}")
            continue

        # Görüntüyü 64x64 boyutuna yeniden boyutlandır
        img = cv2.resize(img, (64, 64))

        # Resmi ve etiketi ilgili dizilere ekle
        images.append(img)
        if directory == directories[0]:
            labels.append("positive")
        elif directory == directories[1]:
            labels.append("negative")

        # Dosya adını eklenen dosyalar setine ekle
        added_files.add(img_path)

# Resim ve etiket dizilerini numpy dizilerine dönüştür
images = np.array(images).reshape(-1, 64, 64, 1)  # Tek kanallı (siyah-beyaz) görüntüler
labels = np.array(labels)




images.shape
labels.shape

labelEn = LabelEncoder() #string olan etiketleri 0 1 2 şeklinde kodla
labels = labelEn.fit_transform(labels)
labels = to_categorical(labels)


#x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.15, shuffle = True,random_state=42)

#np.save('x_train.npy', x_train)
#np.save('y_train.npy', y_train)
#np.save('x_test.npy',  x_test)
#np.save('y_test.npy', y_test)






model = new_model = keras.models.load_model('SMILEs_CNN_AdaptiveLR_Dataset_Increased_80K.h5')

# Modeli değerlendirin
#loss, accuracy = model.evaluate(x_test, y_test)
#print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

predictions = model.predict(images)
print(predictions)

for i in range(len(labels)):
    if predictions[i, 0] > predictions[i, 1]:
        prediction = "negative"
        probability_metric = 0

    elif predictions[i, 0] < predictions[i, 1]:
        prediction = "positive"
        probability_metric = 1

    if labels[i, 1] == 1.0:
        reality = "positive"
    elif labels[i, 1] == 0.0:
        reality = "negative"

    if reality == prediction:
        isCorrect = "Correct!"
    else:
        isCorrect = ""

    cv2.imshow(f"{i}", images[i])
    print(f"Image Number: {i}   Prediction: {prediction} %{round(predictions[i, probability_metric]*100)}    Reality: {reality} {isCorrect}")

cv2.waitKey(0)

print(model.summary())





