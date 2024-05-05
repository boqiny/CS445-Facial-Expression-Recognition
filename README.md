# CS445 - Facial Expression Recognition

This project is the final project for cs445, which combines advanced object detection with facial expression recognition to analyze expressions frame by frame in videos. It utilizes YOLOv9 for robust face detection and a deep learning model using CNN for classifying facial expressions. The workflow includes converting a video into individual frames, detecting faces, classifying their expressions, and then reconstructing the video to view the annotated results.

## Output Video

Watch the output video showing the detected and classified facial expressions by clicking the link below:
[View Output Video](./output_video.mp4)

## Prerequisites

- Python 3.8 or newer
- OpenCV-Python
- TensorFlow 2.x
- Keras
- NumPy
- FFmpeg (for video processing)

## Installation

1. **Clone the repository**:
```
git clone https://github.com/your-repository/csfacial-expression-recognition.git
cd facial-expression-recognition
```

2. **Install required Python libraries**:
```
pip install -r requirements.txt
```

3. **Download and setup YOLOv9**:
- Ensure you have the required weights (`best.pt`) placed in the project directory or adjust the paths in the scripts accordingly.

4. **Prepare the environment**:
- It is recommended to use a virtual environment like conda to avoid conflicts with existing Python packages.

## Usage

### Step 1: Convert Video to Frames
Convert an input video to frames at 30 frames per second. This prepares the video for frame-by-frame analysis.

```
ffmpeg -i input_video.mp4 -vf fps=30 ./frames/output_frame_%04d.png
```

### Step 2: Detect Faces in Frames
Run the face detection script. This script uses a fine-tuned YOLO model to detect faces in each frame.
```
cd ./face_detection/yolo
python detect.py --weights ../../best.pt --source ../../frames/ --save-txt --project runs/detect --name exp_detections
```

### Step 3: Classify Facial Expressions
Use the pretrained expression classification model to classify the expression of each detected face in each frame. (Note that you may need to check the actual input directory in the run folder)
```
cd ../..
python inference.py
```

### Step 4: Convert Frames Back to Video
Convert the processed frames back into a video, maintaining the same frame rate as the original input.

```
ffmpeg -framerate 30 -i ./output_frames/output_frame_%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output_video.mp4
```

## Detection Model Dataset and Training pipeline
Here we used a trained face detection model from yolov9-c.pt which has 25.3M parameters. This is a pretrained yolo model from imagenet and further trained on public human faces detection dataset named WIDER-FACE: http://shuoyang1213.me/WIDERFACE/ over 100 epochs:

First you need to run the following scripts to convert the training dataset to yolo format
```
python train2yolo.py datasets/widerface/train 
python val2yolo.py datasets/widerface
```

To train the model, use
```
python train_dual.py --workers 4 --device 0 --batch 4 --data ../widerface.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights ../yolov9-c.pt --hyp hyp.scratch-high.yaml --min-items 0 --epochs 100 --close-mosaic 15
```

## Classification Model Dataset
FER-2013, RAF-DB

## Contributors
- Boqin Yuan
- Xingyu Qiu

## Reference
https://github.com/WongKinYiu/yolov9

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
