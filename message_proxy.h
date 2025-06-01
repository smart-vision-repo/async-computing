#pragma once
#include <SimpleAmqpClient/SimpleAmqpClient.h>
#include <memory>
#include <string>

class MessageProxy {
public:
  MessageProxy();
  void publishMessage(const std::string &message);
  void sendQueueMessage(const std::string &queue_name,
                        const std::string &message);

private:
  std::string host_;
  int port_;
  std::string user_;
  std::string password_;
  std::string vhost_;
  std::string exchange_;
  std::string routing_key_;

  AmqpClient::Channel::ptr_t channel_;
};