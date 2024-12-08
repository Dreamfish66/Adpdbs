import numpy as np
from sklearn.cluster import dbscan
import pandas as pd


def DBSCANCLU(agents, eps_param, absr, center, d_range, point, learn, nodes_dim=2, max_cluster=10):
    agents_n = len(agents)
    nodes = []
    for agent in agents:
        nodei = [agent.latency[0], agent.Acc[0]]
        nodes.append(nodei)

    nodes = np.array(nodes)
    cluster_ids, cluster_center = dbscan_revise(nodes, d_range, point, eps_param, absr, center, learn)
    #   Cluster draw
    #    df = pd.DataFrame(np.c_[nodes ,cluster_ids],columns = ['feature1','feature2','cluster_id'])
    #    df['cluster_id'] = df['cluster_id'].astype('i2')
    #    df.plot.scatter('feature1','feature2', s = 100,
    #    c = list(df['cluster_id']),cmap = 'rainbow',colorbar = False,
    #    alpha = 0.6,title = 'sklearn DBSCAN cluster result')
    #    print(cluster_ids)
    ids = len(cluster_ids)
    cluster_n = max(cluster_ids)
    clusterednum = 0
    cluster = []
    for i in range(1, cluster_n+1):
        cluster_i = []
        for j in range(ids):
            if cluster_ids[j] == i:
                cluster_i.append(agents[j])
                clusterednum += 1
        cluster.append(cluster_i)
   # print("dbscan clu")
    #print(cluster_ids)
    return cluster, cluster_n, cluster_center, cluster_ids, clusterednum


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def region_query(data, point_idx, eps, labels, cluster_id):
    neighbors = []
    for i in range(len(data)):
        if (euclidean_distance(data[point_idx], data[i]) <= eps and (labels[i] == 0 or labels[i] == cluster_id)):
            neighbors.append(i)
    return neighbors

def max_region(nodes):
    maxregion = [10, 0, 10, 0]
    for node in nodes:
        if node[0] < maxregion[0]:
            maxregion[0] = node[0]
        if node[0] > maxregion[1]:
            maxregion[1] = node[0]
        if node[1] < maxregion[2]:
            maxregion[2] = node[1]
        if node[1] > maxregion[3]:
            maxregion[3] = node[1]
    return maxregion

def check_region(maxregion, node1, absr):
    if ((node1[0] - maxregion[0] < absr) and (maxregion[1] - node1[0] < absr) and
            (node1[1] - maxregion[2] < absr) and (maxregion[3] - node1[1] < absr)):
        return True
    else:
        return False


def expand_cluster(data, labels, point_idx, cluster_id, eps, min_pts, clusteri, maxeps, absr):
    maxregion = max_region(clusteri)
    neighbors = region_query(data, point_idx, eps, labels, cluster_id)
    if len(neighbors) < min_pts:
        labels[point_idx] = -1  #noise
        return False
    else:
        if len(clusteri) > 0:
            cluster_center = np.sum(clusteri) / len(clusteri)
        else:
            cluster_center = data[point_idx]
        centereps = euclidean_distance(cluster_center, data[point_idx])
        if centereps > maxeps:
            return False
        if not check_region(maxregion, data[point_idx], absr):
            return False
        labels[point_idx] = cluster_id
        clusteri.append(data[point_idx])
        maxregion = max_region(clusteri)
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if ((labels[neighbor_idx] == -1) and (euclidean_distance(cluster_center, data[neighbor_idx]) < maxeps)
                    and (check_region(maxregion, data[neighbor_idx], absr))):
                labels[neighbor_idx] = cluster_id
                clusteri.append(data[neighbor_idx])
                maxregion = max_region(clusteri)
            elif ((labels[neighbor_idx] == 0) and (euclidean_distance(cluster_center, data[neighbor_idx]) < maxeps)
                    and (check_region(maxregion, data[neighbor_idx], absr))):
                labels[neighbor_idx] = cluster_id
                clusteri.append(data[neighbor_idx])
                maxregion = max_region(clusteri)
                new_neighbors = region_query(data, neighbor_idx, eps, labels, cluster_id)
                if len(new_neighbors) >= min_pts:
                    neighbors += new_neighbors
            i += 1
        return True

def find_newnodes(nodes, cluster_old):
    centernodes = []
    used = [0] * len(nodes)
    for center in cluster_old:
        distance = euclidean_distance(nodes[0], center)
        center_idx = 0
        for point_idx in range(len(nodes)):
            if euclidean_distance(nodes[point_idx], center) < distance and used[point_idx] == 0:
                center_idx = point_idx
                distance = euclidean_distance(nodes[point_idx], center)
        used[center_idx] = 1
        centernodes.append(center_idx)
    return centernodes

def dbscan_revise(data, eps, min_pts, maxeps, absr, cluster_old, learn):
    labels = [0] * len(data)
    cluster_id = 0
    cluster_center_new = []
    inherited_idx = find_newnodes(data, cluster_old)
    for center_idx in inherited_idx:
        randm = np.random.uniform(0, 1)
        if randm < learn:
            if labels[center_idx] == 0:
                clusteri = []
                if expand_cluster(data, labels, center_idx, cluster_id + 1, eps, min_pts, clusteri, maxeps, absr):
                    cluster_id += 1
                    cluster_center = np.sum(clusteri, 0) / len(clusteri)
                    cluster_center_new.append(cluster_center)
        else:
            point_idx = 0
            while (point_idx < len(data)):
                if labels[point_idx] == 0:
                    clusteri = []
                    if expand_cluster(data, labels, point_idx, cluster_id + 1, eps, min_pts, clusteri, maxeps, absr):
                        cluster_id += 1
                        cluster_center = np.sum(clusteri, 0) / len(clusteri)
                        cluster_center_new.append(cluster_center)
                    break
                else:
                    point_idx += 1

    for point_idx in range(len(data)):
        if labels[point_idx] == 0:
            clusteri = []
            if expand_cluster(data, labels, point_idx, cluster_id + 1, eps, min_pts, clusteri, maxeps, absr):
                cluster_id += 1
                cluster_center = np.sum(clusteri, 0) / len(clusteri)
                cluster_center_new.append(cluster_center)
    return labels, cluster_center_new