#pragma once
#include "models.hpp"
#include <SimpleAmqpClient/SimpleAmqpClient.h>
#include <memory>
#include <string>

class MessageProxy {
public:
  MessageProxy();
  void sendInferResult(const InferenceResult &message);
  void sendDecodeInfo(const TaskDecodeInfo &info);
  void sendInferPackInfo(const TaskInferInfo &info);

private:
  std::string host_;
  int port_;
  std::string user_;
  std::string password_;
  std::string vhost_;
  std::string exchange_;
  std::string routing_key_;
  std::string queue_;
  AmqpClient::Channel::ptr_t channel_;
  void sendMessage(const std::string &message);
};