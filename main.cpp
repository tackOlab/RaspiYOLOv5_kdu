#include <cstdint>
#include <cstdio>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "yolo.hpp"

#if defined(ENABLE_LIBCAMERA)
  #include "LibCamera.h"
#else
  #define INPUT_CAMERA
#endif

constexpr float MODEL_WIDTH  = 160.0f;
constexpr float MODEL_HEIGHT = 160.0f;

int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf("usage: %s class_list modelfile(.onnx) capture-width capture-height\n", argv[0]);
    return EXIT_FAILURE;
  }
  const char *fname_class_list = argv[1];
  const char *onnx_file        = argv[2];

  yolo_class yolo(MODEL_WIDTH, MODEL_HEIGHT);
  // Create a YOLO instance
  try {
    yolo.init(fname_class_list, onnx_file);
  } catch (std::exception &exc) {
    return EXIT_FAILURE;
  }

  assert(yolo.is_empty());

  // Load or capture an image
  cv::Mat frame;
#if defined(INPUT_CAMERA)
  const int32_t cap_width  = std::stoi(argv[3]);
  const int32_t cap_height = std::stoi(argv[4]);
  cv::VideoCapture camera(0);
  if (camera.isOpened() == false) {
    printf("ERROR: cannot open the camera.\n");
    return EXIT_FAILURE;
  }
  camera.set(cv::CAP_PROP_FRAME_WIDTH, cap_width);
  camera.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height);
#elif defined(ENABLE_LIBCAMERA)
  const int32_t cap_width  = std::stoi(argv[3]);
  const int32_t cap_height = std::stoi(argv[4]);
  LibCamera cam;
  int ret = cam.initCamera(cap_width, cap_height, libcamera::formats::RGB888, 4, 0);
  if (ret) {
    printf("ERROR: Failed to initialize camera\n");
    return EXIT_FAILURE;
  }
  libcamera::ControlList controls_;
  /**
   next 2 lines might be problematic with ArduCam 16MP camera (IMX519)
  int64_t frame_time = 1000000 / 30;
  controls_.set(libcamera::controls::FrameDurationLimits, {frame_time, frame_time});
  **/
  cam.set(controls_);
  LibcameraOutData frameData;
  cam.startCamera();
#else
  frame = cv::imread("bus.jpg");
  if (frame.empty()) {
    printf("ERROR: could not find %s.\n", "bus.jpg");
  }
#endif

  cv::Mat img;
  while (true) {
#if defined(INPUT_CAMERA)
    if (camera.read(frame) == false) {
      printf("ERROR: cannot grab a frame\n");
      break;
    }
#elif defined(ENABLE_LIBCAMERA)
    bool flag = cam.readFrame(&frameData);
    if (!flag) continue;
    frame = cv::Mat(cap_height, cap_width, CV_8UC3, frameData.imageData);
#endif

    // Process the image
    yolo.pre_process(frame);
    img = yolo.post_process(frame.clone());

    // Put efficiency information
    double t          = yolo.get_inference_time();
    std::string label = cv::format("Model: %s , Inference time: %6.2f ms", onnx_file, t);
    cv::putText(img, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED, 2);
    imshow("Output", img);

    int32_t keycode = cv::waitKey(1);
    if (keycode == 'q') {
      break;
    }
#if defined(ENABLE_LIBCAMERA)
    cam.returnFrameBuffer(frameData);
#endif
  }
#if defined(ENABLE_LIBCAMERA)
  cam.stopCamera();
  cam.closeCamera();
#endif
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}