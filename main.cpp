#include "message_proxy.h"
#include "video_processor.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

void usage(const char *program_name) {
  std::cerr
      << "Usage: " << program_name
      << " <order_id> <video_file_name> <object_name> <confidence> <interval>\n"
      << "  - order_id: non-empty number\n"
      << "  - video_file_name: non-empty string\n"
      << "  - object_name: non-empty string\n"
      << "  - confidence: float between 0.0 and 1.0\n"
      << "  - interval: positive integer\n"
      << "  - start_frame_index\n";
}

bool validateArguments(int argc, char *argv[]) {
  std::cout << argc << std::endl;
  if (argc != 7) {
    usage(argv[0]);
    return false;
  }

  try {
    int order_id = std::stof(argv[1]);
    float confidence = std::stof(argv[4]);
    int interval = std::stoi(argv[5]);
    int start_frame_index = std::stoi(argv[6]);
    if (argv[1][0] == '\0' || argv[2][0] == '\0' || confidence <= 0.0f ||
        confidence >= 1.0f || interval <= 0) {
      usage(argv[0]);
      return false;
    }
  } catch (const std::exception &e) {
    usage(argv[0]);
    return false;
  }
  return true;
}

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

int main(int argc, char *argv[]) {
  if (!validateArguments(argc, argv)) {
    return 1;
  }
  loadEnvFile();
  int interval = 0;
  int idx = 0;
  std::string order_id = argv[++idx];
  std::string video_file_name = argv[++idx];
  std::string object_name = argv[++idx];
  float confidence = std::stof(argv[++idx]);
  interval = std::stoi(argv[++idx]);
  VideoProcessor processor(order_id, video_file_name, object_name, confidence,
                           interval);
  return processor.process();
}