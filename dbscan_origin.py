import numpy as np
from sklearn.cluster import dbscan


def DBSCANCLU_origin(agents, eps_param, nodes_dim=2, max_cluster=10):
    agents_n = len(agents)
    nodes = []
    for agent in agents:
        nodei = [agent.latency[0], agent.Acc[0]]
        nodes.append(nodei)

    nodes = np.array(nodes)
    cluster_ids = dbscan_revise(nodes, 0.6, 5, eps_param)
    print(cluster_ids)
    ids = len(cluster_ids)
    cluster_n = max(cluster_ids)
    cluster = []
    clustercenter = []
    for i in range(cluster_n + 1):
        cluster_i = []
        acc = 0
        latency = 0
        for j in range(ids):
            if cluster_ids[j] == i:
                cluster_i.append(agents[j])
                acc+=agents[j].Acc
                latency+=agents[j].latency
        if i > 0:
            clustercenter.append([latency / len(cluster_i), acc / len(cluster_i)])
            cluster.append(cluster_i)

    return cluster, cluster_n, clustercenter


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def region_query(data, point_idx, eps):
    neighbors = []
    for i in range(len(data)):
        if euclidean_distance(data[point_idx], data[i]) <= eps:
            neighbors.append(i)
    return neighbors

def expand_cluster(data, labels, point_idx, cluster_id, eps, min_pts, clusteri, maxeps):
    neighbors = region_query(data, point_idx, eps)
    if len(neighbors) < min_pts:
        labels[point_idx] = -1
        return False
    else:
        cluster_center = np.sum(clusteri) / len(clusteri)
        centereps = euclidean_distance(cluster_center, data[point_idx])
        if centereps > maxeps:
            return False
        labels[point_idx] = cluster_id
        clusteri.append(data[point_idx])
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
                clusteri.append(data[neighbor_idx])
            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                clusteri.append(data[neighbor_idx])


                new_neighbors = region_query(data, neighbor_idx, eps)
                if len(new_neighbors) >= min_pts:
                    neighbors += new_neighbors
            i += 1
        return True


def dbscan_revise(data, eps, min_pts, maxeps):
    labels = [0] * len(data)
    cluster_id = 0
    for point_idx in range(len(data)):
        if labels[point_idx] == 0:
            clusteri = []
            if expand_cluster(data, labels, point_idx, cluster_id + 1, eps, min_pts, clusteri, maxeps):
                cluster_id += 1
    return labels