#include "message_proxy.h"
#include <cstdlib>
#include <stdexcept>

MessageProxy::MessageProxy() {
  host_ = std::getenv("RABBITMQ_HOST") ?: "localhost";
  port_ = std::getenv("RABBITMQ_PORT") ? std::stoi(std::getenv("RABBITMQ_PORT"))
                                       : 5672;
  user_ = std::getenv("RABBITMQ_USER") ?: "guest";
  password_ = std::getenv("RABBITMQ_PASS") ?: "guest";
  vhost_ = std::getenv("RABBITMQ_VHOST") ?: "/";
  exchange_ = std::getenv("RABBITMQ_EXCHANGE") ?: "";
  result_queue_ = std::getenv("RABBITMQ_RESULT_QUEUE") ?: "";
  notify_queue_ = std::getenv("RABBITMQ_NOTIFIY_QUEUE") ?: "";

  channel_ =
      AmqpClient::Channel::Create(host_, port_, user_, password_, vhost_);
  channel_->DeclareExchange(
      exchange_, AmqpClient::Channel::EXCHANGE_TYPE_DIRECT, false, true, false);
}

void MessageProxy::sendNotificationMessage(const std::string &message) {
  AmqpClient::BasicMessage::ptr_t msg =
      AmqpClient::BasicMessage::Create(message);
  channel_->BasicPublish("", result_queue_, msg);
}

void MessageProxy::sendInferResultMessage(const std::string &message) {
  AmqpClient::BasicMessage::ptr_t msg =
      AmqpClient::BasicMessage::Create(message);
  channel_->BasicPublish("", notify_queue_, msg);
}

void MessageProxy::sendInferPackInfo(const TaskInferInfo &info) {
  std::ostringstream oss;
  oss << "{"
      << "\"taskId\":\"" << info.taskId << "\","
      << "\"type\":\"" << info.type << "\","
      << "\"completed\":" << info.completed << ","
      << "\"remain\":" << info.remain << "}";
  std::string json_message = oss.str();
  sendNotificationMessage(json_message);
}

void MessageProxy::sendDecodeInfo(const TaskDecodeInfo &info) {
  std::ostringstream oss;
  oss << "{"
      << "\"taskId\":\"" << info.taskId << "\","
      << "\"type\":\"" << info.type << "\","
      << "\"decoded_frames\":" << info.decoded_frames << ","
      << "\"remain_frames\":" << info.remain_frames << "}";
  std::string json_message = oss.str();
  sendNotificationMessage(json_message);
}

void MessageProxy::sendInferResult(const InferenceResult &message) {
  std::ostringstream oss;
  oss << "{"
      << "\"taskId\":\"" << message.taskId << "\","
      << "\"type\":\"" << message.type << "\","
      << "\"frameIndex\":" << message.frameIndex << ","
      << "\"seconds\":" << message.seconds << ","
      << "\"image\":\"" << message.image << "\","
      << "\"confidence\":" << message.confidence << "}";
  std::string json_message = oss.str();
  sendInferResultMessage(json_message);
}
