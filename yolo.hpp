#pragma once

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <opencv2/opencv.hpp>

// Text parameters
constexpr float FONT_SCALE  = 0.7f;
constexpr int32_t FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
constexpr int32_t THICKNESS = 1;

// Colors.
cv::Scalar BLACK  = cv::Scalar(0, 0, 0);
cv::Scalar BLUE   = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED    = cv::Scalar(0, 0, 255);

class yolo_class {
 private:
  const float model_width;
  const float model_height;
  const float score_threshold;
  const float nms_threshold;
  const float confidence_threshold;
  const int32_t rows;
  int32_t af_trigger;
  std::vector<std::string> class_list;
  std::vector<cv::Mat> detections;
  cv::dnn::Net net;
  bool is_set;

 public:
  yolo_class(float w, float h, float th_sc, float th_nms, float th_conf)
      : model_width(w),
        model_height(h),
        score_threshold(th_sc),
        nms_threshold(th_nms),
        confidence_threshold(th_conf),
        // 25200 for default size 640x640
        // 6300 for size 320x320
        // 1575 for size 160x160
        rows((static_cast<int32_t>(model_width) == 160)   ? 1575
             : (static_cast<int32_t>(model_width) == 320) ? 6300
             : (static_cast<int32_t>(model_width) == 640) ? 25200
                                                          : -1),
        is_set(false){};

  ~yolo_class() {
    if (this->is_set) {
      class_list.resize(0);
      detections.resize(0);
      class_list.shrink_to_fit();
      detections.shrink_to_fit();
    }
  }

  void init(const char *fname_class_list, const char *onnx_file) {
    if ((this->rows < 0) || (model_width != model_height)) {
      printf("ERROR: unsupported model size %4.0f x %4.0f\n", model_width, model_height);
      throw std::exception();
    }
    // Load class list
    std::ifstream ifs(fname_class_list);
    if (!ifs) {
      printf("ERROR: could not find %s!\n", fname_class_list);
      throw std::exception();
    }
    std::string line;
    while (getline(ifs, line)) {
      this->class_list.push_back(line);
    }

    // Load model
    try {
      this->net = cv::dnn::readNet(onnx_file);
    } catch (std::exception &exc) {
      printf("ERROR: could not find %s!\n", onnx_file);
      throw std::exception();
    }
    // Set Preferable Backend and Target (currently does not have any effect)
    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    this->is_set = true;
  }

  int32_t get_aftrigger() { return this->af_trigger; }

  bool is_empty() { return this->is_set; }

  inline cv::Mat invoke(cv::Mat &input_image) {
    /****************************************************************************************************
      Pre-process
    ****************************************************************************************************/
    // Convert to blob
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(model_width, model_height), cv::Scalar(),
                           true, false);
    this->net.setInput(blob);
    // Forward propagate
    net.forward(this->detections, getOutputsNames(net)[0]);
    // printf("output size = %d\n", outputs[0].size().area());

    /****************************************************************************************************
     Post-process
    ****************************************************************************************************/
    cv::Mat output_image = input_image.clone();
    std::vector<int32_t> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Resizing factors
    float x_scale  = output_image.cols / model_width;
    float y_factor = output_image.rows / model_height;

    // this number shall be +5 of row number of class list
    const int32_t dimensions = 85;
    // singe row of data_origin consists of;
    // | 0 | 1 | 2 | 3 |      4     | 5 ...                    84|
    // | X | Y | W | H | Confidence | Class scores of 80 clasees |
    float *data_origin = reinterpret_cast<float *>(this->detections[0].data);
    float *p           = data_origin;

    // Iterate through this->row detections
    for (int32_t i = 0; i < this->rows; ++i) {
      float confidence = p[4];
      // Discard bad detections and continue.
      if (confidence >= confidence_threshold) {
        float *classes_scores = p + 5;
        // Create a 1x85 cv::Mat and store class scores of 80 classes
        cv::Mat scores(1, this->class_list.size(), CV_32FC1, classes_scores);
        // Perform minMaxLoc and acquire the index of best class  score
        cv::Point class_id;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
        // Continue if the class score is above the threshold
        if (max_class_score > score_threshold) {
          // Store class ID and confidence in the pre-defined respective vectors
          confidences.push_back(confidence);
          class_ids.push_back(class_id.x);
          // Center
          float cx = p[0];
          float cy = p[1];
          // Box dimension
          float w = p[2];
          float h = p[3];
          // Bounding box coordinates
          int32_t left   = int32_t((cx - 0.5f * w) * x_scale);
          int32_t top    = int32_t((cy - 0.5f * h) * y_factor);
          int32_t width  = int32_t(w * x_scale);
          int32_t height = int32_t(h * y_factor);
          // Store good detections in the boxes vector
          boxes.push_back(cv::Rect(left, top, width, height));
        }
      }
      // Jump to the next row
      p = data_origin + i * dimensions;
    }

    // Perform Non-Maximum Suppression and draw predictions
    std::vector<int32_t> indices;
    bool is_person = false;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold, nms_threshold, indices);
    for (size_t i = 0; i < indices.size(); i++) {
      int32_t idx    = indices[i];
      cv::Rect box   = boxes[idx];
      int32_t left   = box.x;
      int32_t top    = box.y;
      int32_t width  = box.width;
      int32_t height = box.height;
      // Draw bounding box
      cv::rectangle(output_image, cv::Point(left, top), cv::Point(left + width, top + height), BLUE,
                    3 * THICKNESS);
      // Get the label for the class name and its confidence
      std::string label = cv::format("%.2f", confidences[idx]);
      label             = this->class_list[class_ids[idx]] + ":" + label;
      if (class_ids[idx] == 0) {
        is_person = true;
      }
      // Draw class labels
      draw_label(output_image, label, left, top);
    }
    if (is_person) {
      this->af_trigger = 1;
    } else {
      this->af_trigger = 0;
    }
    return output_image;
  }

  std::vector<cv::Mat> &get_detection() { return this->detections; }

  cv::dnn::Net &get_net() { return this->net; }

  double get_inference_time() {
    // The function getPerfProfile returns the overall time for inference(t)
    // and the timings for each of the layers(in layersTimes)
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    return this->net.getPerfProfile(layersTimes) / freq;
  }

 private:
  inline void draw_label(cv::Mat &input_image, std::string label, int32_t left, int32_t top) {
    // Display the label at the top of the bounding box
    int32_t baseLine;
    cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top                 = cv::max(top, label_size.height);
    // Top left corner
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle
    cv::rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
    // Put the label on the black rectangle
    cv::putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW,
                THICKNESS);
  }

  // Get Output Layers Name
  inline std::vector<std::string> getOutputsNames(const cv::dnn::Net &net) {
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
};
