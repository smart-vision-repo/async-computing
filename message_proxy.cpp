#include "message_proxy.h"
#include <cstdlib>
#include <stdexcept>

MessageProxy::MessageProxy() {
    host_ = std::getenv("RABBITMQ_HOST") ?: "localhost";
    port_ = std::getenv("RABBITMQ_PORT") ? std::stoi(std::getenv("RABBITMQ_PORT")) : 5672;
    user_ = std::getenv("RABBITMQ_USER") ?: "guest";
    password_ = std::getenv("RABBITMQ_PASS") ?: "guest";
    vhost_ = std::getenv("RABBITMQ_VHOST") ?: "/";
    exchange_ = std::getenv("RABBITMQ_EXCHANGE") ?: "";
    routing_key_ = std::getenv("RABBITMQ_ROUTING_KEY") ?: "";

    if (exchange_.empty() || routing_key_.empty()) {
        throw std::runtime_error("RABBITMQ_EXCHANGE and RABBITMQ_ROUTING_KEY must be set.");
    }

    channel_ = AmqpClient::Channel::Create(host_, port_, user_, password_, vhost_);
    channel_->DeclareExchange(exchange_, AmqpClient::Channel::EXCHANGE_TYPE_DIRECT, false, true, false);
}

void MessageProxy::sendMessage(const std::string& message) {
    AmqpClient::BasicMessage::ptr_t msg = AmqpClient::BasicMessage::Create(message);
    channel_->BasicPublish(exchange_, routing_key_, msg);
}
