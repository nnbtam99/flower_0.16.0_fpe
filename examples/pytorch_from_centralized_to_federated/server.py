"""Flower server example."""


import flwr as fl

if __name__ == "__main__":
    fl.server.start_server("[::]:8080", config={"num_rounds": 3})
