#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "simple_udp.h"

simple_udp udp0("133.36.41.118", 4001);

int main(int argc, char **argv) {
  FILE *fp = fopen(argv[1], "rb");
  if (fp == nullptr) {
    printf("input file is missing.\n");
    return 1;
  }
  auto fsize = std::filesystem::file_size(argv[1]);
  printf("file size = %lu\n", fsize);
  
  std::vector<uint8_t> data(fsize);
  if (fsize != fread(data.data(), sizeof(uint8_t), fsize, fp)) {
    // error
  };
  fclose(fp);
  auto sizeinstr = std::to_string(fsize);
  udp0.udp_send(sizeinstr);
  udp0.udp_send(data.data(), fsize);
  static const int sendSize = 65000; //UDPの仕様により上限は65kB
  return 0;
}
