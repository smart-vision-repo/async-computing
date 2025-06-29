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
      << "  - object_name: non-empty string\n";
}

bool validateArguments(int argc, char *argv[]) {
  std::cout << argc << std::endl;
  if (argc != 4) {
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
  loadEnvFile();
  if (!validateArguments(argc, argv)) {
    return 1;
  }
  const char* confidenceEnv = std::getenv("CONFIDENCE");
  float confidence = 0.4f;
  if (confidenceEnv) {
    try {
      confidence = std::stof(confidenceEnv);
    } catch (...) {
      std::cerr << "[WARNING] Invalid CONFIDENCE value in .env, using default 0.4\n";
    }
  }
  std::cout << "confidence: " <<  confidence << std::endl;
  int idx = 0;
  int task_id = std::stoi(argv[++idx]);
  std::string video_file_name = argv[++idx];
  std::string object_name = argv[++idx];
  MessageProxy proxy;
  VideoProcessor processor(task_id, video_file_name, object_name, confidence, 30, 0,
                           proxy);
  return processor.process();
}
