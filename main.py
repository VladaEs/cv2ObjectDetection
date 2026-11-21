import cv2
import os

CLASSES = [
    "background", 
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


COLORS = {
    "aeroplane": (255, 0, 0),
    "bicycle": (0, 255, 0),
    "bird": (0, 0, 255),
    "boat": (255, 255, 0),
    "bottle": (255, 0, 255),
    "bus": (0, 255, 255),
    "car": (128, 0, 0),
    "cat": (0, 128, 0),
    "chair": (0, 0, 128),
    "cow": (128, 128, 0),
    "diningtable": (128, 0, 128),
    "dog": (0, 128, 128),
    "horse": (64, 0, 0),
    "motorbike": (0, 64, 0),
    "person": (0, 0, 64),
    "pottedplant": (64, 64, 0),
    "sheep": (64, 0, 64),
    "sofa": (0, 64, 64),
    "train": (192, 192, 192),
    "tvmonitor": (128, 128, 128)
}

prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"
image_path = "images/people.jpg"


#image size
image = cv2.imread(image_path)
h, w = image.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
net = cv2.dnn.readNetFromCaffe(prototxt, model)
net.setInput(blob)
detections = net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence < 0.2:
        continue

    class_id = int(detections[0, 0, i, 1])
    label = CLASSES[class_id]
    box = detections[0, 0, i, 3:7] * [w, h, w, h]
    (startX, startY, endX, endY) = box.astype("int")  

    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(w - 1, endX)
    endY = min(h - 1, endY)
    text = "{}: {:.2f}%".format(label, confidence * 100)
    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS.get(label, (0, 0, 0)), 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(label, (0, 0, 0)), 2)

cv2.imshow("DNN Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()