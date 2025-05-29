## dependencies:

### RabbitMQ
- MacOS
    ```sh 
    # 安装 Boost
    brew install boost

    # 安装 RabbitMQ C client
    brew install rabbitmq-c

    # 安装 SimpleAmqpClient
    brew install simple-amqp-client
    ```
- Ubuntu
    ```sh
    sudo apt update
    sudo apt install -y libboost-all-dev librabbitmq-dev
    # 安装构建依赖
    sudo apt install -y cmake g++ git

    # 克隆 SimpleAmqpClient
    git clone https://github.com/alanxz/SimpleAmqpClient.git
    cd SimpleAmqpClient

    # 构建并安装
    cmake .
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    ``` 