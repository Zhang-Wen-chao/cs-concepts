CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

all: server client

server: rpc_server.cpp example_server.cpp
	$(CXX) $(CXXFLAGS) rpc_server.cpp example_server.cpp -o server

client: rpc_client.cpp example_client.cpp
	$(CXX) $(CXXFLAGS) rpc_client.cpp example_client.cpp -o client

clean:
	rm -f server client

.PHONY: all clean
