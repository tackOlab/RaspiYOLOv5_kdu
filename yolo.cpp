#include <cstdint>
#include <cstdio>
#include <fstream>
#include <opencv2/opencv.hpp>

#if defined(ENABLE_LIBCAMERA)
  #include "LibCamera.h"
#else
  #define INPUT_CAMERA
#endif

// Constants.
constexpr float MDEL_WIDTH           = 160.0f;
constexpr float MODEL_HEIGHT         = 160.0f;
constexpr float SCORE_THRESHOLD      = 0.5f;
constexpr float NMS_THRESHOLD        = 0.45f;
constexpr float CONFIDENCE_THRESHOLD = 0.45f;

// Text parameters.
constexpr float FONT_SCALE  = 0.7f;
constexpr int32_t FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
constexpr int32_t THICKNESS = 1;

// Colors.
cv::Scalar BLACK  = cv::Scalar(0, 0, 0);
cv::Scalar BLUE   = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED    = cv::Scalar(0, 0, 255);

void draw_label(cv::Mat &input_image, std::string label, int32_t left, int32_t top) {
  // Display the label at the top of the bounding box.
  int32_t baseLine;
  cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
  top                 = cv::max(top, label_size.height);
  // Top left corner.
  cv::Point tlc = cv::Point(left, top);
  // Bottom right corner.
  cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
  // Draw white rectangle.
  cv::rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
  // Put the label on the black rectangle.
  cv::putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW,
              THICKNESS);
}

// Get Output Layers Name
std::vector<std::string> getOutputsNames(const cv::dnn::Net &net) {
  static std::vector<std::string> names;
  if (names.empty()) {
    std::vector<int32_t> out_layers       = net.getUnconnectedOutLayers();
    std::vector<std::string> layers_names = net.getLayerNames();
    names.resize(out_layers.size());
    for (size_t i = 0; i < out_layers.size(); ++i) {
      names[i] = layers_names[out_layers[i] - 1];
    }
  }
  return names;
}

std::vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net) {
  // Convert to blob.
  cv::Mat blob;
  cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(MDEL_WIDTH, MODEL_HEIGHT), cv::Scalar(),
                         true, false);

  net.setInput(blob);

  // Forward propagate.
  std::vector<cv::Mat> outputs;
  // old implemantation
  // net.forward(outputs, net.getUnconnectedOutLayersNames());
  net.forward(outputs, getOutputsNames(net)[0]);
  // printf("output size = %d\n", outputs[0].size().area());
  return outputs;
}

// Initialize vectors to hold respective outputs while unwrapping detections.
cv::Mat post_process(cv::Mat &&input_image, std::vector<cv::Mat> &outputs,
                     const std::vector<std::string> &class_name) {
  std::vector<int32_t> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  // Resizing factors
  float x_scale  = input_image.cols / MDEL_WIDTH;
  float y_factor = input_image.rows / MODEL_HEIGHT;

  // this number shall be +5 of row number of class list
  const int32_t dimensions = 85;
  // singe row of data_origin consists of;
  // | 0 | 1 | 2 | 3 |      4     | 5 ...                    84|
  // | X | Y | W | H | Confidence | Class scores of 80 clasees |
  float *data_origin = reinterpret_cast<float *>(outputs[0].data);
  float *p           = data_origin;

  const int32_t rows = 1575;  // 25200 for default size 640, 6300 for 320x320,
                              // 1575 for 160x160
  // Iterate through 25200 detections.
  for (size_t i = 0; i < rows; ++i) {
    float confidence = p[4];
    // Discard bad detections and continue.
    if (confidence >= CONFIDENCE_THRESHOLD) {
      float *classes_scores = p + 5;
      // Create a 1x85 cv::Mat and store class scores of 80 classes.
      cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
      // Perform minMaxLoc and acquire the index of best class  score.
      cv::Point class_id;
      double max_class_score;
      minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
      // Continue if the class score is above the threshold.
      if (max_class_score > SCORE_THRESHOLD) {
        // Store class ID and confidence in the pre-defined respective vectors.
        confidences.push_back(confidence);
        class_ids.push_back(class_id.x);
        // Center.
        float cx = p[0];
        float cy = p[1];
        // Box dimension.
        float w = p[2];
        float h = p[3];
        // Bounding box coordinates.
        int32_t left   = int32_t((cx - 0.5f * w) * x_scale);
        int32_t top    = int32_t((cy - 0.5f * h) * y_factor);
        int32_t width  = int32_t(w * x_scale);
        int32_t height = int32_t(h * y_factor);
        // Store good detections in the boxes vector.
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
    // Jump to the next row.
    p = data_origin + i * dimensions;
  }

  // Perform Non-Maximum Suppression and draw predictions.
  std::vector<int32_t> indices;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
  for (size_t i = 0; i < indices.size(); i++) {
    int32_t idx    = indices[i];
    cv::Rect box   = boxes[idx];
    int32_t left   = box.x;
    int32_t top    = box.y;
    int32_t width  = box.width;
    int32_t height = box.height;
    // Draw bounding box.
    cv::rectangle(input_image, cv::Point(left, top), cv::Point(left + width, top + height), BLUE,
                  3 * THICKNESS);
    // Get the label for the class name and its confidence.
    std::string label = cv::format("%.2f", confidences[idx]);
    label             = class_name[class_ids[idx]] + ":" + label;
    // Draw class labels.
    draw_label(input_image, label, left, top);
  }
  return input_image;
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf("usage: %s class_list modelfile(.onnx) capture-width capture-height\n", argv[0]);
    return EXIT_FAILURE;
  }
  const char *fname_class_list = argv[1];
  const char *onnx_file        = argv[2];
  // Load class list.
  std::vector<std::string> class_list;
  std::ifstream ifs(fname_class_list);
  if (!ifs) {
    printf("ERROR: could not find %s!\n", fname_class_list);
    return EXIT_FAILURE;
  }
  std::string line;
  while (getline(ifs, line)) {
    class_list.push_back(line);
  }
  // Load model.
  cv::dnn::Net net;
  try {
    net = cv::dnn::readNet(onnx_file);
  } catch (std::exception &exc) {
    printf("ERROR: could not find %s!\n", onnx_file);
    return EXIT_FAILURE;
  }
  // Set Preferable Backend and Target (currently does not have any effect)
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

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
  int64_t frame_time = 1000000 / 30;
  controls_.set(libcamera::controls::FrameDurationLimits, {frame_time, frame_time});
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
    std::vector<cv::Mat> detections;
    // Process the image
    detections = pre_process(frame, net);
    img        = post_process(frame.clone(), detections, class_list);

    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t)
    // and the timings for each of the layers(in layersTimes)
    std::vector<double> layersTimes;
    double freq       = cv::getTickFrequency() / 1000;
    double t          = net.getPerfProfile(layersTimes) / freq;
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