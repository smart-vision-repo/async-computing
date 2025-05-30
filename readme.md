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

### TensorRT

```sh
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb

sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-*/nv-tensorrt-*.key /usr/share/keyrings/

sudo apt update
sudo apt install tensorrt
sudo apt install libnvinfer-dev libnvinfer8
sudo apt install libnvonnxparsers-dev libnvonnxparsers8
sudo apt install libnvinfer-plugin-dev libnvinfer-plugin8

```