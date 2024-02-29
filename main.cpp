#include "yolo.hpp"
#include <chrono>
#include <ctime>
#include "ohtj2k_codec.h"

#if defined(ENABLE_LIBCAMERA)
  #include "LibCamera.h"
#endif

#include "simple_udp.hpp"
simple_udp udp0("133.36.41.118", 4001);

// Define the size of model
// 1. width and height shall be equal to each ohter
// 2. shall be equal to 160 or 320 or 640
// Corresponding .onnx is required!
constexpr float MODEL_WIDTH  = 160.0f;
constexpr float MODEL_HEIGHT = 160.0f;

// Constants
constexpr float SCORE_THRESHOLD      = 0.5f;
constexpr float NMS_THRESHOLD        = 0.45f;
constexpr float CONFIDENCE_THRESHOLD = 0.45f;

int main(int argc, char *argv[]) {
  htj2k::Encoder *htenc;
  htenc = new htj2k::OHTJ2KEncoder();
  htj2k::CodestreamBuffer cb;

  struct timespec ts;
  struct tm tmstruct;

  std::chrono::_V2::high_resolution_clock::time_point start;
  if (argc != 3 && argc != 5) {
    printf("usage: %s class_list modelfile(.onnx) <capture-width capture-height>\n", argv[0]);
    return EXIT_FAILURE;
  }
  const char *fname_class_list = argv[1];
  const char *onnx_file        = argv[2];

  bool INPUT_CAMERA = (argc == 5) ? true : false;

  int32_t tmpw = 640, tmph = 480;
  if (INPUT_CAMERA) {
    tmpw = std::stoi(argv[3]);
    tmph = std::stoi(argv[4]);
  }
  const int32_t cap_width  = tmpw;
  const int32_t cap_height = tmph;

  yolo_class yolo(MODEL_WIDTH, MODEL_HEIGHT, SCORE_THRESHOLD, NMS_THRESHOLD, CONFIDENCE_THRESHOLD);
  // Create a YOLO instance
  try {
    yolo.init(fname_class_list, onnx_file);
  } catch (std::exception &exc) {
    return EXIT_FAILURE;
  }

  assert(yolo.is_empty());

  // Load or capture an image
  cv::Mat frame;
#if defined(ENABLE_LIBCAMERA)
  LibCamera cam;
  int ret = cam.initCamera(cap_width, cap_height, libcamera::formats::RGB888, 4, 0);
  if (ret) {
    printf("ERROR: Failed to initialize camera\n");
    return EXIT_FAILURE;
  }
  libcamera::ControlList controls_;
  int64_t frame_time = 1000000 / 30;
  // Set frame rate
  controls_.set(libcamera::controls::FrameDurationLimits,
                libcamera::Span<const int64_t, 2>({frame_time, frame_time}));
  // Adjust the brightness of the output images, in the range -1.0 to 1.0
  controls_.set(libcamera::controls::Brightness, 0.0);
  // Adjust the contrast of the output image, where 1.0 = normal contrast
  controls_.set(libcamera::controls::Contrast, 1.0);
  // Set the exposure time
  //  controls_.set(libcamera::controls::ExposureTime, 20000);
  // Set Auto exposure
  controls_.set(libcamera::controls::AeEnable, libcamera::controls::AE_ENABLE);
  // Set autofocus mode
  controls_.set(libcamera::controls::AfMode, libcamera::controls::AfModeContinuous);
  controls_.set(libcamera::controls::AfMetering, libcamera::controls::AfMeteringAuto);
  controls_.set(libcamera::controls::AfRange, libcamera::controls::AfRangeNormal);
  controls_.set(libcamera::controls::AfSpeed, libcamera::controls::AfSpeedNormal);

  cam.set(controls_);
  LibcameraOutData frameData;
  cam.startCamera();
#else
  cv::VideoCapture camera(0);
  if (INPUT_CAMERA) {
    if (camera.isOpened() == false) {
      printf("ERROR: cannot open the camera.\n");
      return EXIT_FAILURE;
    }
    camera.set(cv::CAP_PROP_FRAME_WIDTH, cap_width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height);
  } else {
    frame = cv::imread("bus.jpg");
    if (frame.empty()) {
      printf("ERROR: could not find %s.\n", "bus.jpg");
      return EXIT_FAILURE;
    }
  }
#endif
  cv::Mat output_image;
  bool isAFstable = false;
  while (true) {
#if defined(ENABLE_LIBCAMERA)
    bool flag = cam.readFrame(&frameData);
    if (!flag) continue;
    frame = cv::Mat(cap_height, cap_width, CV_8UC3, frameData.imageData);
#else
    if (INPUT_CAMERA) {
      if (camera.read(frame) == false) {
        printf("ERROR: cannot grab a frame\n");
        break;
      }
    }
#endif
    int tr0 = yolo.get_aftrigger();

    // Process the image
    output_image = yolo.invoke(frame);

    int tr1 = yolo.get_aftrigger();

    // Put efficiency information
    double t          = yolo.get_inference_time();
    std::string label = cv::format("Model: %s , Inference time: %6.2f ms", onnx_file, t);
    cv::putText(output_image, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED, 2);
    imshow("Output", output_image);

    int32_t keycode = cv::waitKey(1);

    if (keycode == 'q') {
      break;
    }
#if defined(ENABLE_LIBCAMERA)
    if (tr1 && (tr0 != tr1)) {
      controls_.set(libcamera::controls::AfTrigger, libcamera::controls::AfTriggerStart);
      cam.set(controls_);
      start      = std::chrono::high_resolution_clock::now();
      isAFstable = false;  // autofocus is not stable
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    auto count    = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    double time   = static_cast<double>(count) / 1000.0;

    if (time > 2600.0) {
      isAFstable = true;  // autofocus is stable
    }

    char tbuf[32];
    auto now          = std::chrono::high_resolution_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm    = *std::localtime(&now_c);
    // strftime(tbuf, 32, "%F-%T", &now_tm);
    strftime(tbuf, 32, "%Y-%m-%d-%H-%M-%S", &now_tm);

    std::timespec_get(&ts, TIME_UTC);
    char tmbuf[64];
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::nanoseconds{ts.tv_nsec});
    const int msec = ms.count();
    snprintf(tmbuf, 64, "%s.%03d", tbuf, msec);

    const std::vector<int> Quality = {90};
    if (isAFstable && tr1) {
      std::string fname = cv::format("%s.j2c", tmbuf);
      cv::cvtColor(frame, output_image, cv::COLOR_BGR2RGB);
      cb           = htenc->encodeRGB8(output_image.data, output_image.cols, output_image.rows);
      size_t csize = cb.size;
      udp0.udp_send(std::to_string(csize));
      udp0.udp_send(cb.codestream, csize);
      // FILE *fp = fopen(fname.c_str(), "wb");
      // fwrite(cb.codestream, sizeof(uint8_t), cb.size, fp);

      // fclose(fp);
    }

    cam.returnFrameBuffer(frameData);
#endif
  }

#if defined(ENABLE_LIBCAMERA)
  cam.returnFrameBuffer(frameData);
  cam.stopCamera();
  cam.closeCamera();
#endif
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
