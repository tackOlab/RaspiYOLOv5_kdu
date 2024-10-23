#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>

#define BUFFER_MAX 4096

class simple_udp {
  int sock;
  struct sockaddr_in addr;

 public:
  simple_udp(std::string address, int port) {
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(address.c_str());
    addr.sin_port = htons(port);
    // int bufsz = BUFFER_MAX;
    // setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char *)&bufsz, sizeof(bufsz));
  }

  void udp_send(std::string word) {
    sendto(sock, word.c_str(), word.length(), 0, (struct sockaddr *)&addr,
           sizeof(addr));
  }

  void udp_send(uint8_t *p, size_t length) {
    sendto(sock, p, length, 0, (struct sockaddr *)&addr, sizeof(addr));
  }

  void udp_bind() {
    bind(sock, (const struct sockaddr *)&addr, sizeof(addr));
    int flags = fcntl(sock, F_GETFL);
    int result = fcntl(sock, F_SETFL, flags | O_NONBLOCK);
  }

  std::string udp_recv() {
    char buf[BUFFER_MAX];
    memset(buf, 0, sizeof(buf));
    recv(sock, buf, sizeof(buf), 0);
    return std::string(buf);
  }

  void udp_recv(char *buf, size_t size) {
    memset(buf, 0, sizeof(buf));
    recv(sock, buf, size, 0);
    // printf("%ld\n", length);
    // if (length > 0) {
    //   for (auto i = 0; i < length; ++i) {
    //     dst[i] = buf[i];
    //   }
    // }
    // return length;
  }

  ~simple_udp() { close(sock); }
};
