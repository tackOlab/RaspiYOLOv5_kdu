# RaspiYOLOv5
Sample of object detection using YOLOv5 on Raspberry pi with cameras.



## Build instructions

```
cmake -B./build -DCMAKE_BUILDTYPE=Release -DCMAKE_CXX_COMPILER=/path/to/your-compiler
cd build
make
cd bin
```

## Command line sage

Example for object detection using a `yolo5n.onnx` model with captured images having size of 640x480 (width x height)

```
./yolo ../../coco.names ../../yolov5n.onnx 640 480
```
