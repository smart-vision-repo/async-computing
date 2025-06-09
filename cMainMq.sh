export CPATH=$(brew --prefix)/include
export LIBRARY_PATH=$(brew --prefix)/lib
g++ -o test_mq test_mq.cpp message_proxy.cpp -lSimpleAmqpClient -lboost_system -lpthread $(pkg-config --cflags --libs opencv4)
