from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image_path = "images/test.jpg"
img = cv2.imread(image_path)

results = model(img)

for r in results:
    annotated = r.plot()

cv2.imshow("Detected Image", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
