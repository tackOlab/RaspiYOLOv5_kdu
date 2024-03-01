#pragma once
#include <chrono>
#include <ctime>
#include <string>

std::string create_filename_based_on_time() {
  char tbuf[32];
  auto now          = std::chrono::high_resolution_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm now_tm    = *std::localtime(&now_c);
  strftime(tbuf, 32, "%Y-%m-%d-%H-%M-%S", &now_tm);

  struct timespec ts;
  std::timespec_get(&ts, TIME_UTC);
  char tmbuf[64];
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::nanoseconds{ts.tv_nsec});
  const int msec = ms.count();
  snprintf(tmbuf, 64, "%s.%03d.j2c", tbuf, msec);
  std::string fname(tmbuf);
  return fname;
}