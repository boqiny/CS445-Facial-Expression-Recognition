import cv2
import numpy as np
from keras.models import load_model
from pathlib import Path

# Load the expression classification model
model_expression = load_model('model_custom.h5')

def predict_label(model, image):
    num2label = {
        0: 'Surprise',
        1: 'Fear',
        2: 'Disgust',
        3: 'Happy',
        4: 'Sad',
        5: 'Angry',
        6: 'Neutral'
    }
    expression_pred = model.predict(image)
    expression = np.argmax(expression_pred)
    return num2label[expression]


def process_images(image_dir, bbox_dir, output_dir):
    files = sorted(Path(image_dir).glob('*.png'))
    for file in files:
        image = cv2.imread(str(file))
        h, w = image.shape[:2]
        bbox_file = Path(bbox_dir) / (file.stem + '.txt')

        with open(bbox_file, 'r') as f:
            bboxes = f.readlines()

        for bbox in bboxes:
            parts = bbox.strip().split()
            x_center, y_center, width, height = map(float, parts[1:5])
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            face_crop = image[y1:y2, x1:x2]
            face_crop = cv2.resize(face_crop, (100, 100))  # Resize to the input size of pretrained expression classification model
            face_crop = face_crop.astype('float32') / 255.0  # Normalize the pixel values
            face_crop = np.expand_dims(face_crop, axis=0)  # Add a batch dimension

            expression_label = predict_label(model_expression, face_crop)  

            # Draw rectangle around the face
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put the expression label below the bounding box
            text_size = cv2.getTextSize(expression_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = x1
            text_y = y2 + 10 + text_size[1]
            cv2.putText(image, expression_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path / file.name), image)


image_dir = './face_detection/yolo/runs/detect/exp_detections'
bbox_dir = './face_detection/yolo/runs/detect/exp_detections/labels' 
output_dir = 'output_frames'
process_images(image_dir, bbox_dir, output_dir)
