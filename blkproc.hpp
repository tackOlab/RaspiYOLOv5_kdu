#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <opencv2/opencv.hpp>

constexpr int DCTSIZE = 8;

void blkproc(cv::Mat &in, std::function<void(cv::Mat &, int *)> func, int *p = nullptr) {
  for (int y = 0; y < in.rows; y += DCTSIZE) {
    for (int x = 0; x < in.cols; x += DCTSIZE) {
      cv::Mat blk_in, blk_out;
      blk_in  = in(cv::Rect(x, y, DCTSIZE, DCTSIZE)).clone();
      blk_out = in(cv::Rect(x, y, DCTSIZE, DCTSIZE));
      func(blk_in, p);
      blk_in.convertTo(blk_out, blk_out.type());
    }
  }
}

namespace blk {
void dct2(cv::Mat &in, int *dummy) {
  if (in.rows != DCTSIZE || in.cols != DCTSIZE || in.channels() != 1) {
    printf("input for block_dct2() shall be 8x8 and monochrome.\n");
    exit(EXIT_FAILURE);
  }
  cv::dct(in, in);
}

void mask(cv::Mat &in, int *dummy) {
  if (in.rows != DCTSIZE || in.cols != DCTSIZE || in.channels() != 1) {
    printf("input for block_dct2() shall be 8x8 and monochrome.\n");
    exit(EXIT_FAILURE);
  }
  float *p = (float *)in.data;
  // clang-format off
  float mask[] = {
    0, 0, 0, 0, 0, 0, 1, 1, 
    0, 0, 0, 0, 0, 1, 1, 1, 
    0, 0, 0, 0, 1, 1, 1, 1, 
    0, 0, 0, 1, 1, 1, 1, 1, 
    0, 0, 1, 1, 1, 1, 1, 1, 
    0, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1
  };
  // clang-format on
  float sum = 0.0;
  for (int i = 0; i < in.rows * in.cols; ++i) {
    p[i] *= mask[i];
    if (p[i] < 0.0) {
      p[i] = -p[i];
    }
    sum += p[i];
  }
  sum /= DCTSIZE * DCTSIZE;
  for (int i = 0; i < in.rows * in.cols; ++i) {
    p[i] = sum;
  }
}
}  // namespace blk