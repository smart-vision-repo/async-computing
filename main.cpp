#include "video_processor.h"
#include <filesystem>
#include <iostream>

void usage(const char *program_name) {
  std::cerr << "Usage: " << program_name
            << " <video_file_name> <object_name> <confidence> <interval>\n"
            << "  - video_file_name: non-empty string\n"
            << "  - object_name: non-empty string\n"
            << "  - confidence: float between 0.0 and 1.0\n"
            << "  - interval: positive integer\n";
}

bool validateArguments(int argc, char *argv[]) {
  if (argc != 5) {
    usage(argv[0]);
    return false;
  }

  try {
    float confidence = std::stof(argv[3]);
    int interval = std::stoi(argv[4]);

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

int main(int argc, char *argv[]) {
  if (!validateArguments(argc, argv)) {
    return 1;
  }
  int interval = 0;
  std::string video_file_name = argv[1];
  std::string object_name = argv[2];
  float confidence = std::stof(argv[3]);
  interval = std::stoi(argv[4]);
  VideoProcessor processor(video_file_name, object_name, confidence, interval);
  return processor.process();
}