
#include "message_proxy.h"
#include <fstream>
#include <iostream>
#include <string>

void loadEnvFile() {
  const std::vector<std::string> possiblePaths = {
      ".env", std::string(getenv("HOME")) + "/.env"};
  std::ifstream envFile;
  for (const auto &path : possiblePaths) {
    envFile.open(path);
    if (envFile.is_open()) {
      std::string line;
      while (std::getline(envFile, line)) {
        std::istringstream lineStream(line);
        std::string key, value;
        if (std::getline(lineStream, key, '=') &&
            std::getline(lineStream, value)) {
          if (!key.empty() && !value.empty()) {
            setenv(key.c_str(), value.c_str(), 1);
          }
        }
      }
      envFile.close();
      return;
    }
  }
  std::cerr << "[WARNING] .env file not found in current or user directory."
            << std::endl;
}

int main() {
  try {
    loadEnvFile();
    MessageProxy proxy;

    // Test sendInferInfo
    TaskInferInfo inferInfo;
    inferInfo.taskId = 11;
    inferInfo.gFrameIndex = 10;
    inferInfo.seconds = 5.0;
    inferInfo.results = 1;
    proxy.sendInferInfo(inferInfo);
    std::cout << "sendInferInfo passed with taskId: " << inferInfo.taskId
              << std::endl;

    // Test sendDecodeInfo
    TaskDecodeInfo decodeInfo;
    decodeInfo.taskId = 12;
    decodeInfo.decoded_frames = 100;
    decodeInfo.remain_frames = 50;
    proxy.sendDecodeInfo(decodeInfo);
    std::cout << "sendDecodeInfo passed with taskId: " << decodeInfo.taskId
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
