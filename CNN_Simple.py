import os
import cv2
import numpy as np
#import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import matplotlib.pyplot as plt


# Sınıf isimlerini ve giriş yollarını tanımla
directories = ["SMILEs/positives/positives7", "SMILEs/negatives/negatives7"]

# Resim ve etiket dizilerini oluştur
images = []
labels = []

added_files = set()

for directory in directories:
    for file_name in os.listdir(directory):
        img_path = f"{directory}/{file_name}"

        if file_name in added_files:
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
        added_files.add(file_name)

# Resim ve etiket dizilerini numpy dizilerine dönüştür
images = np.array(images).reshape(-1, 64, 64, 1)  # Tek kanallı (siyah-beyaz) görüntüler
labels = np.array(labels)




images.shape
labels.shape

labelEn = LabelEncoder() #string olan etiketleri 0 1 2 şeklinde kodla
labels = labelEn.fit_transform(labels)
labels = to_categorical(labels)


x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = .15, shuffle = True,random_state=42)

np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy',  x_test)
np.save('y_test.npy', y_test)






model = Sequential()
model.add(Conv2D(64, (7, 7), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((5, 5)))

model.add(Flatten())

model.add(Dense(2, activation='softmax'))  # 2 sınıflı çıkış katmanı

# Modeli derleyin
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin
training = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Modeli değerlendirin
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")



model.save('SMILEs_CNN_Simple.h5')

print(model.summary())

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(training.history['loss'], label='train loss')
plt.plot(training.history['val_loss'], label='test loss')
plt.title('train and test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(training.history['accuracy'], label='train accuarcy')
plt.plot(training.history['val_accuracy'], label='test accuracy')
plt.title('train and test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



