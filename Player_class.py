import numpy as np
import random


class Framework:
    def __init__(self, properties, cost, index, Name="YOLOv5"):
        self.pro_len = len(properties)
        self.Latency = properties[0]
        self.Acc = properties[1]
        self.Computation = properties[2]
        self.Communication = properties[3]
        self.Index = index
        self.Name = Name


class Client:
    def __init__(self, properties):
        self.character_length = len(properties)
        self.latency = properties[0]
        self.Acc = properties[1]
        self.latency_lie = properties[2]
        self.Acc_lie = properties[3]
        self.Model = []
        self.lie = 0
        self.value = []
        self.value_lie = []

    def get_evaluation(self, frameworks):
        for framework in frameworks:
            self.value.append(framework.Latency * self.latency + framework.Acc * self.Acc)
            self.value_lie.append(framework.Latency * self.latency_lie + framework.Acc * self.Acc_lie)
            self.Model.append(framework.Name)
        return self.value


class Server:
    def __init__(self, properties):
        self.character_length = len(properties)
        self.computation = properties[0]
        self.communication = properties[1]
        self.Model = []
        self.cost = []

    def get_evaluation(self, frameworks):
        for framework in frameworks:
            self.cost.append(framework.Computation * self.computation + framework.Communication * self.communication)
            self.Model.append(framework.Name)
        return self.cost
