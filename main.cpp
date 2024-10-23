#include <cassert>

#include <HTJ2KEncoder.hpp>
#include <cstring>
#include "yolo.hpp"
#include <opencv2/highgui.hpp>
#include "LibCamera.h"
#include "simple_tcp.hpp"
#include "create_filename.hpp"

#include "model_config.hpp"

/* ========================================================================= */
/*                         Set up messaging services                         */
/* ========================================================================= */

class kdu_stream_message : public kdu_core::kdu_thread_safe_message
{
public: // Member classes
    kdu_stream_message(std::ostream *stream)
    {
        this->stream = stream;
    }
    void put_text(const char *string)
    {
        (*stream) << string;
    }
    void flush(bool end_of_message = false)
    {
        stream->flush();
        kdu_thread_safe_message::flush(end_of_message);
    }

private: // Data
    std::ostream *stream;
};

static kdu_stream_message cout_message(&std::cout);
static kdu_stream_message cerr_message(&std::cerr);
static kdu_core::kdu_message_formatter pretty_cout(&cout_message);
static kdu_core::kdu_message_formatter pretty_cerr(&cerr_message);

/*************************************************************************************************/
// MAIN
/*************************************************************************************************/
int main(int argc, char *argv[]) {
  // cv::setNumThreads(0);

  if (argc != 4 && argc != 6) {
    printf("usage: %s class_list modelfile(.onnx) <Qfactor> <capture-width capture-height> \n", argv[0]);
    return EXIT_FAILURE;
  }
  const char *fname_class_list = argv[1];
  const char *onnx_file        = argv[2];

  bool INPUT_CAMERA = (argc == 6) ? true : false;

  const int Quality = std::stoi(argv[3]);
  int32_t tmpw = 640, tmph = 480;
  if (INPUT_CAMERA) {
    tmpw = std::stoi(argv[4]);
    tmph = std::stoi(argv[5]);
  }
  const int32_t cap_width  = tmpw;
  const int32_t cap_height = tmph;

  HTJ2KEncoder encoder;
  enum progression {
    LRCP, RLCP, RPCL, PCRL, CPRL
  };
  encoder.setQuality(false, 0.0f);
  encoder.setDecompositions(5);
  encoder.setBlockDimensions(Size(64, 64));
  encoder.setProgressionOrder(RPCL);
  encoder.setQfactor(Quality);
  
  const FrameInfo info = {static_cast<uint16_t>(cap_width), static_cast<uint16_t>(cap_height), 8, 3, false};
  std::vector<uint8_t> &rawBytes = encoder.getDecodedBytes(info);
  rawBytes.resize(0);
  rawBytes.reserve(cap_width * cap_height * 3);

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
  controls_.set(libcamera::controls::AfMode, libcamera::controls::AfModeAuto);
  controls_.set(libcamera::controls::AfMetering, libcamera::controls::AfMeteringAuto);
  controls_.set(libcamera::controls::AfRange, libcamera::controls::AfRangeNormal);
  controls_.set(libcamera::controls::AfSpeed, libcamera::controls::AfSpeedNormal);
  //   Set manual focus mode
  // controls_.set(libcamera::controls::AfMode, libcamera::controls::AfModeManual);
  // controls_.set(libcamera::controls::LensPosition, 0.5);
  cam.set(controls_);  // Write camera settings

  LibcameraOutData frameData;
  cam.startCamera();

  cv::Mat output_image;

  while (true) {  // loop begin
    bool flag = cam.readFrame(&frameData);
    if (!flag) continue;
    frame = cv::Mat(cap_height, cap_width, CV_8UC3, frameData.imageData);

    __attribute__((unused)) int tr0 = yolo.get_aftrigger();

    // Object detection by YOLOv5
    output_image = yolo.invoke(frame);

    int tr1 = yolo.get_aftrigger();

    // Put efficiency information
    double t          = yolo.get_inference_time();
    std::string label_yolo = cv::format("Model: %s , Inference time: %6.2f ms", onnx_file, t);
    cv::putText(output_image, label_yolo, cv::Point(20, 40), FONT_FACE, FONT_SCALE, WHITE, 2);
    
    int32_t keycode = cv::pollKey();

    if (keycode == 'q') {
      break;
    }

    /*************************************************************************************************/
    // HTJ2K encoding
    /*************************************************************************************************/
    std::string label_htj2k = cv::format("");
    if (tr1 || keycode == 'c') {
      cv::Mat RGBimg;
      std::string fname = create_filename_based_on_time();
      cv::cvtColor(frame, RGBimg, cv::COLOR_BGR2RGB);
      encoder.setSourceImage(RGBimg.data, RGBimg.cols * RGBimg.rows * 3);
      auto t_j2k_0 = std::chrono::high_resolution_clock::now();
      encoder.encode();
      auto t_j2k    = std::chrono::high_resolution_clock::now() - t_j2k_0;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_j2k).count();
      std::vector cb = encoder.getEncodedBytes();
      label_htj2k = cv::format("HT Encoding takes %6.2f [ms], codestream size = %zu bytes",
             static_cast<double>(duration) / 1000.0, cb.size());
      cv::putText(output_image, label_htj2k, cv::Point(20, cap_height - 60), FONT_FACE, FONT_SCALE, WHITE, 2);
      // send codestream via TCP connection
      simple_tcp tcp_socket("133.36.41.118", 4001);
      tcp_socket.create_client();
      tcp_socket.Tx(cb.data(), cb.size());
    }

    /*************************************************************************************************/
    // Get temperature
    /*************************************************************************************************/
    FILE *fp;
    char temp_read[16]="";
    float pi_temp;
    fp = fopen("/sys/class/thermal/thermal_zone0/temp","r");
    fgets(temp_read, 16, fp);
    fclose(fp);
    pi_temp = std::stoi(temp_read, nullptr, 10) / 1000.0f;
    cv::putText(output_image, cv::format("temp = %6.2f 'C", pi_temp), cv::Point(cap_width - 200, cap_height - 60), FONT_FACE, FONT_SCALE, WHITE, 2);

    // Show image
    imshow("Output", output_image);

    cam.returnFrameBuffer(frameData);
  }  // loop end

  cam.returnFrameBuffer(frameData);
  cam.stopCamera();
  cam.closeCamera();

  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
