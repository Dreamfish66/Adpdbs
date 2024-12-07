import numpy as np
import random
import os
from sklearn.cluster import KMeans

os.environ['OMP_NUM_THREADS'] = '1'

def KMEANSLU(agents, beta_param, nodes_dim=2, max_cluster=10):
    agents_n = len(agents)
    nodes = []
    for agent in agents:
        nodei = [agent.latency[0], agent.Acc[0]]
        nodes.append(nodei)

    nodes = np.array(nodes)
    kmeans = KMeans(n_clusters=8, random_state=0, n_init=10)
    kmeans.fit(nodes)
    cluster_ids = kmeans.labels_
    #print(cluster_ids)
    ids = len(cluster_ids)
    cluster_n = max(cluster_ids) + 1
    cluster = []
    cluster_center = []
    for i in range(cluster_n):
        cluster_i = []
        cluster_centeri = []
        for j in range(ids):
            if cluster_ids[j] == i:
                cluster_i.append(agents[j])
                cluster_centeri.append(nodes[j])
        cluster.append(cluster_i)
        cluster_center.append(np.sum(cluster_centeri, 0) / len(cluster_centeri))
    return cluster, cluster_n, cluster_center

def randomcluster(agents, beta_param, nodes_dim=2, max_cluster=10):
    agents_n = len(agents)
    cluster_n = 4
    ifpick = np.ones(agents_n, dtype=bool)
    cluster = []
    cluster_center = []
    for i in range(cluster_n):
        node = []
        if i != cluster_n-1:
            numbern = agents_n // cluster_n
        else:
            numbern = agents_n - (agents_n // cluster_n) * i
        cluster_i = []
        while numbern > 0:
            pick = random.randint(0, agents_n-1)
            if ifpick[pick]:
                numbern -= 1
                cluster_i.append(agents[pick])
                node.append([agents[pick].latency, agents[pick].Acc])
                ifpick[pick] = False
        cluster.append(cluster_i)
        cluster_center.append(np.sum(node, 0) / len(node))
    #print(cluster_center)
    return cluster, cluster_n, cluster_center

def noncluster(agents, beta_param, nodes_dim=2, max_cluster=10):
    agents_n = len(agents)
    cluster_n = 1
    ifpick = np.ones(agents_n, dtype=bool)
    cluster = []
    cluster_center = []
    for i in range(cluster_n):
        node = []
        if i != cluster_n-1:
            numbern = agents_n // cluster_n
        else:
            numbern = agents_n - (agents_n // cluster_n) * i
        cluster_i = []
        while numbern > 0:
            pick = random.randint(0, agents_n-1)
            if ifpick[pick]:
                numbern -= 1
                cluster_i.append(agents[pick])
                node.append([agents[pick].latency, agents[pick].Acc])
                ifpick[pick] = False
        cluster.append(cluster_i)
        cluster_center.append(np.sum(node, 0) / len(node))
    print("noncluster center", cluster_center)
    return cluster, cluster_n, cluster_center, 1


