import cv2
import numpy as np

cap = cv2.VideoCapture(0)                     # 啟用攝影鏡頭
print('loading...')
# 載入模型
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf

model = Sequential()
model.add(Conv2D(32, (5,5), activation="relu", padding="same", data_format="channels_last", input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
model.add(Conv2D(32, (5,5), activation="relu", padding="same", data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.load_weights('cnn2.weights.h5')
print('start...')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
try:
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img = cv2.resize(img,(540,300))          # 改變影像尺寸，加快處理效率
        x, y, w, h = 400, 180, 120, 120          # 定義擷取數字的區域位置和大小，寬高加倍，並調整 Y 座標以避免超出影像範圍
        img_num = img.copy()                     # 複製一個影像作為辨識使用
        img_num = img_num[y:y+h, x:x+w]          # 擷取辨識的區域

        img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)    # 顏色轉成灰階
        # 針對白色文字，做二值化黑白轉換，轉成黑底白字
        ret, img_num = cv2.threshold(img_num, 127, 255, cv2.THRESH_BINARY_INV)
        output = cv2.cvtColor(img_num, cv2.COLOR_GRAY2BGR)     # 顏色轉成彩色
        img[0:120, 420:540] = output                           # 將轉換後的影像顯示在畫面右上角，尺寸為 120x120

        img_num = cv2.resize(img_num,(28,28))   # 縮小成 28x28，和訓練模型對照
        img_num = img_num.astype(np.float32)    # 轉換格式
        img_num = np.expand_dims(img_num, axis=0) # 增加一個維度作為 batch size
        img_num = np.expand_dims(img_num, axis=3) # 增加一個維度作為 channel
        img_num = img_num/255
        img_pre = model.predict(img_num)        # 進行辨識
        num = str(np.argmax(img_pre))           # 取得辨識結果

        text = num                              # 印出的文字內容
        org = (x,y-20)                          # 印出的文字位置
        fontFace = cv2.FONT_HERSHEY_SIMPLEX     # 印出的文字字體
        fontScale = 2                           # 印出的文字大小
        color = (0,0,255)                       # 印出的文字顏色
        thickness = 2                           # 印出的文字邊框粗細
        lineType = cv2.LINE_AA                  # 印出的文字邊框樣式
        cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType) # 印出文字

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)  # 標記辨識的區域
        cv2.imshow('oxxostudio', img)
        key = cv2.waitKey(1) # 調整等待時間，確保視窗能正常顯示
        if key == ord('q'): # 按下 q 鍵停止
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
