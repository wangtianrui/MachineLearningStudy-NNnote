import cv2
import os

font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体

image = cv2.imread(r"./testdata/diamodtest.jpg")

zuoshangx = 20
zuoshangy = 10
youxiax = 90
youxiay = 700

cv2.rectangle(image, (int(zuoshangx), int(zuoshangy)), (int(youxiax), int(youxiay)), (0, 225, 0), 3)
# 图片，左上角，右下角，颜色，线条粗细，点类型

if (zuoshangy > 10):
    #防止框在上部文字无法放置
    cv2.putText(image, "test", (int(zuoshangx), int(zuoshangy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 225, 0))
else:
    cv2.putText(image, "test", (int(zuoshangx), int(zuoshangy + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 225, 0))

cv2.imshow("xixi", image)
cv2.waitKey(0)
