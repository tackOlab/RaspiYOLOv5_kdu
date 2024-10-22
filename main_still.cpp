#include <cstdint>
#include <cstdio>
#include <string>
#include <cassert>
#include "yolo.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "model_config.hpp"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::printf("usage: %s class_list modelfile(.onnx)\n", argv[0]);
    return EXIT_FAILURE;
  }
  const char *fname_class_list = argv[1];
  const char *onnx_file        = argv[2];

  yolo_class yolo(MODEL_WIDTH, MODEL_HEIGHT, SCORE_THRESHOLD, NMS_THRESHOLD, CONFIDENCE_THRESHOLD);
  // Create a YOLO instance
  try {
    yolo.init(fname_class_list, onnx_file);
  } catch (std::exception &exc) {
    return EXIT_FAILURE;
  }

  assert(yolo.is_empty());

  // Load  an image
  cv::Mat frame;
  frame = cv::imread("../../bus.jpg");
  if (frame.empty()) {
    std::printf("ERROR: could not find %s.\n", "bus.jpg");
    return EXIT_FAILURE;
  }

  cv::Mat output_image;
  while (true) {
    // Process the image
    output_image = yolo.invoke(frame);

    // Put efficiency information
    double t          = yolo.get_inference_time();
    std::string label = cv::format("Model: %s , Inference time: %6.2f ms", onnx_file, t);
    cv::putText(output_image, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED, 2);
    cv::imshow("Output", output_image);

    int32_t keycode = cv::waitKey(1);

    if (keycode == 'q') {
      break;
    }
  }

  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
