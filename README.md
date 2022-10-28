# RaspiYOLOv5
Sample of object detection using YOLOv5 on Raspberry pi with cameras.

# Prerequisites

## Basic Python environment
- Python 3.7 or higher
- pip

## YOLOv5

Clone the official repository.

```
git clone https://github.com/ultralytics/yolov5.git
```

### Setup to convert models in `.pt` to `.onnx` form

1. Install required packages by pip

```
pip install -r requirements.txt
```

2. Install pytorch version 1.11 and torchvision version 0.12.0 and onnx

```
pip install torch==1.11 torchvision==0.12.0 onnx
```

3. If any, install utils

```
pip install utils
```

4. Convert model files

On the top directory of the cloned YOLOv5, try command-line like follows.

```
python export.py --data data/coco128.yaml --weights yolov5n.pt --include onnx --img 160 160
```

- Note 1: yolov5n.pt can be other model's name (e.g. yolo5s.pt, yolo5m.pt, etc)
- Note 2: Two intergers followed by --img are width and height of the model. The smaller values give the better throughput but the lower precision.

## Build instructions

```
cmake -B./build -DCMAKE_BUILDTYPE=Release -DCMAKE_CXX_COMPILER=/path/to/your-compiler
cd build
make
cd bin
```

## Command line usage

Example for object detection using the converted `yolo5n.onnx` model with captured images having size of 640x480 (width x height)

```
./yolo ../../coco.names ../../yolov5n.onnx 640 480
```
