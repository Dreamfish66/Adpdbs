import numpy as np
import networkx as nx
import random
import math
from Player_class import *
from VCG_implement import *
from UKmeans import *
from dbscan import *
from kmeans import *
from dbscan_origin import *


def Init_entity(Num_per_cluster, Center_node_list, framework_list, datatype):
    Client_list = []
    Server_List = []
    index = 0
    abnormal_probability = 0.1
    for center in Center_node_list:
        index += 1
        for i in range(Num_per_cluster):
            if datatype == "line":
                random_move1 = np.random.uniform(-0.5, 0.5, 1)
                random_move2 = np.random.normal(0, 0.3, 1)
                Cli0 = center[0] + random_move1 * 3
                Cli1 = center[1] + random_move1 * 3 + random_move2 * 3
                Cli0_lie = Cli0 + np.random.normal(0, 0.3, 1) * Cli0
                Cli1_lie = Cli1 + np.random.normal(0, 0.3, 1) * Cli1
                number = random.uniform(0, 1)
                # Random abnormal points
                if (number < abnormal_probability):
                    Cli0 = center[0] + np.random.uniform(-0.2, 0.2, 1) * random.uniform(50, 100)
                    Cli1 = center[1] + np.random.uniform(-0.2, 0.2, 1) * random.uniform(50, 100)
                    Cli0_lie = Cli0 + np.random.normal(0, 0.3, 1) * Cli0
                    Cli1_lie = Cli1 + np.random.normal(0, 0.3, 1) * Cli1
                Cli_properties = [Cli0, Cli1, Cli0_lie, Cli1_lie]
            else:
                Cli0 = center[0] + np.random.normal(0, 0.4, 1) * 2.5
                Cli1 = center[1] + np.random.normal(0, 0.4, 1) * 2.5
                Cli0_lie = Cli0 + np.random.normal(0, 0.15, 1) * Cli0
                Cli1_lie = Cli1 + np.random.normal(0, 0.15, 1) * Cli1
                number = random.uniform(0, 1)
                # Random abnormal points
                if (number < abnormal_probability):
                    Cli0 = center[0] + np.random.uniform(-0.2, 0.2, 1) * random.uniform(50, 100)
                    Cli1 = center[1] + np.random.uniform(-0.2, 0.2, 1) * random.uniform(50, 100)
                    Cli0_lie = Cli0 + np.random.normal(0, 0.3, 1) * Cli0
                    Cli1_lie = Cli1 + np.random.normal(0, 0.3, 1) * Cli1
                Cli_properties = [Cli0, Cli1, Cli0_lie, Cli1_lie]
            Ser0 = center[0] + np.random.uniform(-0.2, 0.2, 1) * 5
            Ser1 = center[1] + np.random.uniform(-0.2, 0.2, 1) * 5
            Ser_properties = [Ser0, Ser1]
            client = Client(properties=Cli_properties)
            client.get_evaluation(framework_list)
            server = Server(properties=Ser_properties)
            server.get_evaluation(framework_list)
            Client_list.append(client)
            Server_List.append(server)
    random.shuffle(Client_list)
    random.shuffle(Server_List)

    return Client_list, Server_List

class Crowdfunding_env:
    def __init__(self, Num_per_cluster, Center_node_list, framework_list, datatype):
        self.Num_per_cluster = Num_per_cluster
        self.Center_node_list = Center_node_list
        self.num_agent = len(Center_node_list) * Num_per_cluster
        self.framework_list = framework_list
        self.datatype = datatype
        self.Client_list, self.Server_list= Init_entity(Num_per_cluster, Center_node_list,
                                                        self.framework_list, self.datatype)
        self.num_model = len(self.framework_list)

        # self.Relation_Matrix = np.zeros([self.num_agent, self.num_agent])
        # self.Reward_Matrix = np.zeros([self.num_agent, self.num_agent])
        # self.Indicator_Matrix = np.zeros([self.num_agent, self.num_agent])
        # self.Graph = nx.from_numpy_array(self.Relation_Matrix)
        #self.Graph = Agent_G(self.num_agent)

        self.Total_social_welfare = []
        self.Total_client_utility = []
        self.cli_cluster = []
        self.cli_cluster_num = 1
        self.ser_cluster = []
        self.ser_cluster_num = 1

        self.chosen_models = []
        self.social_utilitys = []
        self.clientutilityfalse = 0
        self.client_taxs = []
        self.server_taxs = []
        self.cluster_center = []

        self.beta = 1
        self.maxeps = 3
        self.absmaxr = 7
        self.learn = 50
        self.dbscan_range = 0.7
        self.dbscan_point = 5
        self.cluster_ids = []
        self.clustered_num = 0

    def printcluster(self):
        latency_list = []
        acc_list = []
        for client in self.Client_list:
                latency_list.append(client.latency[0])
                acc_list.append(client.Acc[0])
        with open('cluster_data.py', 'a') as f:
            f.write(f"latency_list = {latency_list}\n")
            f.write(f"acc_list = {acc_list}\n")
            f.write(f"cluster_list = {self.cluster_ids}\n")
    def Evolve(self):
        # Evolution of the environment and adaptation
        if self.clientutilityfalse > 0:
            self.beta *= 0.95
            self.maxeps *= 0.99
            self.absmaxr *= 0.99
        for centernode in self.Center_node_list:
            centernode = [centernode[0] + 0.01, centernode[1] + 0.01]

    def reset(self):

        #self.Client_list, self.Server_list = Init_entity(self.Num_per_cluster, self.Center_node_list,
        #                                                self.framework_list, self.datatype)
        for client in self.Client_list:
            client.latency += np.random.normal(0, 0.01)
            client.Acc += np.random.normal(0, 0.01)
        self.num_model = len(self.framework_list)

        # self.Relation_Matrix = np.zeros([self.num_agent, self.num_agent])
        # self.Reward_Matrix = np.zeros([self.num_agent, self.num_agent])
        # self.Indicator_Matrix = np.zeros([self.num_agent, self.num_agent])
        # self.Graph = nx.from_numpy_array(self.Relation_Matrix)
        #self.Graph = Agent_G(self.num_agent)

        self.Total_social_welfare = []
        self.Total_client_utility = []
        self.cli_cluster = None
        self.cli_cluster_num = 1
        self.ser_cluster = None
        self.ser_cluster_num = 1

        self.chosen_models = []
        self.social_utilitys = []
        self.clientutilityfalse = 0
        self.client_taxs = []
        self.server_taxs = []

    def generate_graph(self):
        for i in range(self.num_agent):
            for j in range(self.num_agent):
                self.Relation_Matrix[i][j] = 0.5 * ((self.Client_list[i].Acc - self.Client_list[j].Acc) ** 2
                                                    + (self.Client_list[i].latency - self.Client_list[j].latency) ** 2)

        self.Graph = nx.from_numpy_array(self.Relation_Matrix)
        self.Graph.edges(data=True)

        return self.Graph

    def cluster(self, Agents, algorithm, isGraph=False):
        Agent_cluster = None
        Cluster_num = 1
        if algorithm == "ukmean":
            Agent_cluster, Cluster_num, self.cluster_center = UK_mean(Agents, self.beta)
            #print("ukmean")
        elif algorithm == "kmeans":
            Agent_cluster, Cluster_num, self.cluster_center = KMEANSLU(Agents, self.beta)
            #print("kmeans")
        elif algorithm == "ukmeanlie":
            Agent_cluster, Cluster_num = UK_meanlie(Agents, self.beta)
            #print("ukmeanlie")
        elif algorithm == "random":
            Agent_cluster, Cluster_num, self.cluster_center = randomcluster(Agents, self.beta)
        elif algorithm == "non":
            Agent_cluster, Cluster_num, self.cluster_center, self.clustered_num = noncluster(Agents, self.beta)
            #print("random")\
        elif algorithm == "dbscan_origin":
            Agent_cluster, Cluster_num, self.cluster_center = DBSCANCLU_origin(Agents, self.maxeps)
        else:
            Agent_cluster, Cluster_num, self.cluster_center, self.cluster_ids, self.clustered_num = DBSCANCLU(Agents, self.maxeps, self.absmaxr,
                                                                        self.cluster_center, self.dbscan_range,
                                                                        self.dbscan_point, self.learn)
            #print("DBSCAN")

        #if isGraph:
        #    if algorithm == "louvain":
        #        Agent_cluster, Cluster_num = Louvain(Agents)
        #    elif algorithm == "leiden":
        #        Agent_cluster, Cluster_num = Leiden(Agents)
        #
        #else:
        #    if algorithm == "ukmean":
        #        Agent_cluster, Cluster_num = Ukmean(Agents, self.beta)
        #    elif algorithm == "meanshift":
        #        Agent_cluster, Cluster_num = Meanshift(Agents)

        return Agent_cluster, Cluster_num

    def client_cluster(self):
        self.cli_cluster, self.cli_cluster_num = self.cluster(self.Client_list, algorithm="louvain")
        return self.cli_cluster, self.cli_cluster_num

    def server_cluster(self):
        self.ser_cluster, self.ser_cluster_num = self.cluster(self.Server_list, algorithm="louvain")
        return self.ser_cluster, self.ser_cluster_num

    def get_server_list(self):
        model_cost = []
        for server_list in self.ser_cluster:
            cost = []
            for server in server_list:
                cost.append(server.cost)
            model_cost.append(cost)
        return model_cost

    def get_client_list(self):
        model_value = []
        for client_list in self.cli_cluster:
            value = []
            for client in client_list:
                value.append(client.value)
            model_value.append(value)
        return model_value

    def get_client_list_show(self):
        model_value = []
        for client_list in self.cli_cluster:
            value = []
            for client in client_list:
                if client.lie == 1:
                    value.append(client.value_lie)
                else:
                    value.append(client.value)
            model_value.append(value)
        return model_value

    def get_client_list_extracost(self):
        model_value = []
        clu_n = 0
        for client_list in self.cli_cluster:
            value = []
            for client in client_list:
                valuei = []
                #print(self.cluster_center)
                for framework in self.framework_list:
                    valuei.append(framework.Latency * np.abs(client.latency - self.cluster_center[clu_n][0])
                                 + framework.Acc * np.abs(client.Acc - self.cluster_center[clu_n][1]))
                value.append(valuei)
            model_value.append(value)
            clu_n += 1
        return model_value



    def crowdfunding(self):
        client_lists = self.get_client_list()
        server_lists = self.get_server_list()
        extracost_lists = self.get_client_list_extracost()

        # assume one server_cluster for simplisity
        server_list = server_lists[0]
        cluster_cost = 0
        cluster_value = 0
        reward_relax = 0
        reward_nonrelax = 0
        non_utility = 0
        negative_num = 0

        for i in range(self.cli_cluster_num):
            winner_model, social_utility, client_tax, server_tax, bid_negative, cluster_costi, cluster_valuei, r_relax, r_nlrelax, bid_negative_num= VCG_auction(client_lists[i], server_list, extracost_lists[i])
            self.chosen_models.append(winner_model)
            self.social_utilitys.append(social_utility)
            self.client_taxs.append(client_tax)
            self.server_taxs.append(server_tax)
            self.clientutilityfalse += bid_negative
            if (bid_negative > 0):
                non_utility += np.sum(social_utility)
            else: non_utility += np.sum(social_utility)
            cluster_value += cluster_valuei
            cluster_cost += cluster_costi
            reward_relax += r_relax
            reward_nonrelax += r_nlrelax
            negative_num += bid_negative_num
        return self.chosen_models, self.social_utilitys, self.client_taxs, self.server_taxs, cluster_cost, cluster_value, reward_relax, reward_nonrelax, non_utility, negative_num

    def crowdfunding_lie(self):
        client_lists = self.get_client_list()
        #print("Client value")
        #print(client_lists)
        server_lists = self.get_server_list()
        #print("Server value")
        #print(server_lists)

        # assume one server_cluster for simplisity
        server_list = server_lists[0]

        client_lists_show = self.get_client_list_show()

        for i in range(self.cli_cluster_num):
            winner_model, social_utility, client_tax, server_tax, bid_negative, client_utility = VCG_auction_lie(client_lists_show[i], server_list, client_lists[i])
            self.chosen_models.append(winner_model)
            self.social_utilitys.append(social_utility)
            self.client_taxs.append(client_tax)
            self.server_taxs.append(server_tax)
            self.clientutilityfalse += bid_negative

        return self.chosen_models, self.social_utilitys, self.client_taxs, self.server_taxs, client_utility[self.luck]

    def crowdfunding_uncluster(self):
        client_lists = self.get_client_list()
        #print("Client value")
        #print(client_lists)
        server_lists = self.get_server_list()
        #print("Server value")
        #print(server_lists)
        extracost_lists = self.get_client_list_extracost()

        # assume one server_cluster for simplisity
        client_list = client_lists[0]
        server_list = server_lists[0]
        extracost_lists = extracost_lists[0]

        winner_model, social_utility, client_tax, server_tax, bid_negative, client_utility = VCG_auction(client_list, server_list, extracost_lists)
        self.chosen_models.append(winner_model)
        self.social_utilitys.append(social_utility)
        self.client_taxs.append(client_tax)
        self.server_taxs.append(server_tax)
        self.clientutilityfalse += bid_negative
        #print("luck:", self.luck)
        #print("no lie", client_list[self.luck])
        return self.chosen_models, self.social_utilitys, self.client_taxs, self.server_taxs, client_utility[self.luck]

    def crowdfunding_uncluster_lie(self):
        client_lists = self.get_client_list()
        #print("Client value")
        #print(client_lists)
        server_lists = self.get_server_list()
        #print("Server value")
        #print(server_lists)
        extracost_lists = self.get_client_list_extracost_lie()

        # assume one server_cluster for simplisity
        client_list = client_lists[0]
        server_list = server_lists[0]

        client_lists_show = self.get_client_list_show()
        client_lists_show = client_lists_show[0]
        extracost_lists = extracost_lists[0]
        cl = 0

        winner_model, social_utility, client_tax, server_tax, bid_negative, client_utility = VCG_auction_lie(client_lists_show, server_list, client_list, extracost_lists)
        self.chosen_models.append(winner_model)
        self.social_utilitys.append(social_utility)
        self.client_taxs.append(client_tax)
        self.server_taxs.append(server_tax)
        self.clientutilityfalse += bid_negative
        #print("luck:", self.luck)
        #print("lie", client_lists_show[self.luck])
        #print("no lie", client_list[self.luck])
        return self.chosen_models, self.social_utilitys, self.client_taxs, self.server_taxs, client_utility[self.luck][0]
