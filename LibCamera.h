#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/libcamera.h>
#include <libcamera/property_ids.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>
#include <libcamera/transform.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// using namespace libcamera;

typedef struct {
  uint8_t *imageData;
  uint32_t size;
  uint64_t request;
} LibcameraOutData;

class LibCamera {
 public:
  LibCamera(){};
  ~LibCamera(){};

  int initCamera(int width, int height, libcamera::PixelFormat format,
                 int buffercount, int rotation);

  int startCamera();
  bool readFrame(LibcameraOutData *frameData);
  void returnFrameBuffer(LibcameraOutData frameData);

  void set(libcamera::ControlList controls);
  void stopCamera();
  void closeCamera();

 private:
  int startCapture();
  int queueRequest(libcamera::Request *request);
  void requestComplete(libcamera::Request *request);
  void processRequest(libcamera::Request *request);

  unsigned int cameraIndex_;
  uint64_t last_;
  std::unique_ptr<libcamera::CameraManager> cm;
  std::shared_ptr<libcamera::Camera> camera_;
  bool camera_acquired_ = false;
  bool camera_started_ = false;
  std::unique_ptr<libcamera::CameraConfiguration> config_;
  std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
  std::vector<std::unique_ptr<libcamera::Request>> requests_;
  // std::map<std::string, Stream *> stream_;
  std::map<int, std::pair<void *, unsigned int>> mappedBuffers_;

  std::queue<libcamera::Request *> requestQueue;

  libcamera::ControlList controls_;
  std::mutex control_mutex_;
  std::mutex camera_stop_mutex_;
  std::mutex free_requests_mutex_;
};