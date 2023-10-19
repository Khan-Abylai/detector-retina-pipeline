import cv2
# 283KZY05_13-18-28.jpeg



x1, y1, w, h, lx1, ly1, _, lx2, ly2, _, lx3, ly3, _, lx4, ly4, _, lx5, ly5, _, _ = 161.00000381469727, 666.0000085830688, 386.9999885559082, 135.9999704360962, 161.00000381469727, \
                                                                                   666.0000085830688, 0.0, 547.9999923706055, 727.0000076293945, 0.0, 354.0000057220459, 733.999993801117, \
                                                                                   0.0, 161.00000381469727, 740.0000095367432, 0.0, 547.9999923706055, 801.999979019165, 0.0, 0.8
img = cv2.imread("../7257f17f05d5dd7e13a917e15d557b06.jpg")

cv2.circle(img, (int(x1), int(y1)), 3, (0, 0, 255), -1)
cv2.putText(img, "left top", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

cv2.circle(img, (int(lx1), int(ly1)), 3, (255, 0, 0), -1)
cv2.putText(img, "1", (int(lx1), int(ly1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

cv2.circle(img, (int(lx2), int(ly2)), 3, (0, 0, 0), -1)
cv2.putText(img, "2", (int(lx2), int(ly2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

cv2.circle(img, (int(lx3), int(ly3)), 3, (0, 255, 0), -1)
cv2.putText(img, "3", (int(lx3), int(ly3 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

cv2.circle(img, (int(lx4), int(ly4)), 3, (0, 0, 255), -1)
cv2.putText(img, "4", (int(lx4), int(ly4 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

cv2.circle(img, (int(lx5), int(ly5)), 3, (255, 255, 0), -1)
cv2.putText(img, "5", (int(lx5), int(ly5 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
cv2.imwrite("../7257f17f05d5dd7e13a917e15d557b06_label.jpg", img)