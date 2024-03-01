#include <cassert>

#include "ohtj2k_codec.h"
#include "yolo.hpp"
#include "LibCamera.h"
#include "blkproc.hpp"
#include "simple_udp.hpp"
#include "create_filename.hpp"

#include "model_config.hpp"

void DCT_driven_focusing(cv::Mat &in, uint8_t &count_bit, bool &isAFstable, float &past_average) {
  cv::Mat f0 = in.clone();
  cv::cvtColor(f0, f0, cv::COLOR_BGR2GRAY, 1);
  f0.convertTo(f0, CV_32F);
  blkproc(f0, blk::dct2);  // 2D 8x8 DCT
  blkproc(f0, blk::mask);  // mask lower frequency out
  cv::resize(f0, f0, cv::Size(), 1.0 / DCTSIZE, 1.0 / DCTSIZE, cv::INTER_NEAREST);
  cv::imshow("high", f0);

  float *f0p = (float *)f0.data;
  float a0   = 0.0;
  for (int i = 0; i < f0.rows * f0.cols; ++i) {
    a0 += f0p[i];
  }
  a0 /= f0.rows * f0.cols;
  float diff = a0 - past_average;
  count_bit <<= 1;
  if (diff < 0.0) {
    count_bit |= 1;
  }
  if (isAFstable == false) {
    printf("%f, %f\n", a0, diff);
  }
  count_bit &= 0x7;
  if (count_bit == 0x7) {
    isAFstable = true;
  }
  past_average = a0;
}

int main(int argc, char *argv[]) {
  simple_udp udp_sock("133.36.41.118", 4001);

  htj2k::Encoder *htenc;
  htenc = new htj2k::OHTJ2KEncoder();

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
  LibCamera cam;
  int ret = cam.initCamera(cap_width, cap_height, libcamera::formats::RGB888, 4, 0);
  if (ret) {
    printf("ERROR: Failed to initialize camera\n");
    return EXIT_FAILURE;
  }
  libcamera::ControlList controls_;
  int64_t frame_time = 1000000 / 30;
  /**************************************************************************************/
  // Camera settings
  /**************************************************************************************/
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
  // Focus related settings
  //   Set autofocus mode
  // controls_.set(libcamera::controls::AfMode, libcamera::controls::AfModeContinuous);
  // controls_.set(libcamera::controls::AfMetering, libcamera::controls::AfMeteringAuto);
  // controls_.set(libcamera::controls::AfRange, libcamera::controls::AfRangeNormal);
  // controls_.set(libcamera::controls::AfSpeed, libcamera::controls::AfSpeedNormal);
  //   Set manual focus mode
  controls_.set(libcamera::controls::AfMode, libcamera::controls::AfModeManual);
  controls_.set(libcamera::controls::LensPosition, 0);
  cam.set(controls_);  // Write camera settings

  LibcameraOutData frameData;
  cam.startCamera();

  cv::Mat output_image;
  bool isAFstable   = false;
  uint8_t count_bit = 0;
  float LENSPOS     = 0.0;
  // LENSPOS
  // 0 moves the lens to infinity.
  // 0.5 moves the lens to focus on objects 2m away.
  // 2 moves the lens to focus on objects 50cm away.
  // And larger values will focus the lens closer.
  float past_average = 0.0;
  float max_average  = 0.0;
  float MAX_POS      = 0.0;

  bool focus_fixed = false;
  while (true) {  // loop begin
    bool flag = cam.readFrame(&frameData);
    if (!flag) continue;
    frame = cv::Mat(cap_height, cap_width, CV_8UC3, frameData.imageData);

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

    // if (tr1 && (tr0 != tr1)) {
    //   controls_.set(libcamera::controls::AfTrigger, libcamera::controls::AfTriggerStart);
    //   cam.set(controls_);
    //   start      = std::chrono::high_resolution_clock::now();
    //   isAFstable = false;  // autofocus is not stable

    // if (keycode == 'f') {
    //   LENSPOS -= 0.05;
    // }
    // if (keycode == 'c') {
    //   LENSPOS += 0.05;
    // }
    // controls_.set(libcamera::controls::LensPosition, LENSPOS);
    // cam.set(controls_);
    // printf("LENPOS = %f\n", LENSPOS);

    start = std::chrono::high_resolution_clock::now();
    if (!focus_fixed) {
      DCT_driven_focusing(frame, count_bit, isAFstable, past_average);
      if (max_average < past_average) {
        max_average = past_average;
        MAX_POS     = LENSPOS;
      }

      if (isAFstable == false) {
        // MOVE LENS (CHANGE FOCUS)
        LENSPOS = LENSPOS + 0.1;
        controls_.set(libcamera::controls::LensPosition, LENSPOS);
        cam.set(controls_);
      } else {
        controls_.set(libcamera::controls::LensPosition, MAX_POS);
        cam.set(controls_);
        focus_fixed = true;
      }
    }

    auto duration = std::chrono::high_resolution_clock::now() - start;
    auto count    = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    double time   = static_cast<double>(count) / 1000.0;
    printf("Focusing takes %6.2f [ms]\n", time);

    // if (time > 2600.0) {
    //   isAFstable = true;  // autofocus is stable
    // }

    const std::vector<int> Quality = {90};
    if (isAFstable && tr1) {
      std::string fname = create_filename_based_on_time();
      cv::cvtColor(frame, output_image, cv::COLOR_BGR2RGB);
      htj2k::CodestreamBuffer cb =
          htenc->encodeRGB8(output_image.data, output_image.cols, output_image.rows);
      udp_sock.udp_send(std::to_string(cb.size));
      udp_sock.udp_send(cb.codestream, cb.size);
    }
    cam.returnFrameBuffer(frameData);
  }  // loop end

  cam.returnFrameBuffer(frameData);
  cam.stopCamera();
  cam.closeCamera();

  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
