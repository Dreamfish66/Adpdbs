import numpy as np
import pandas as pd

class node:
    def __init__(self, properties,model_n, value=100):
        self.latency = properties[0]
        self.Acc = properties[2]
        self.value = np.random.normal(value, value/10, size=model_n)


def get_u(nodes, nodes_n, cluster_cen, cluster_n, gamma, alpha, rate):
    """
    :param nodes:Data of point(x_i)
    :param nodes_n:the number of nodes in total
    :param cluster_cen:a_k, the culture center k
    :param cluster_n:C, the number of clusters
    :param gamma: parameter, e^(-c/450)
    :param alpha:the list of alpha_t, the probability of one data point belonged to the kth class
    :param rate:
    :return:u_t+1
    """
    u = np.zeros((nodes_n, cluster_n))
    difference = np.zeros((nodes_n, cluster_n))
    for k in range(cluster_n):
        # Get the difference of k column
        difference_k = np.sum(np.square(nodes - cluster_cen[k]), 1) - gamma * np.log(alpha[k])
        # Add the column k to difference
        for i in range(nodes_n):
            difference[i][k] = difference_k[i]
    if rate == 1:
        np.fill_diagonal(difference, np.nan)  # Forbidden a note use itself as cluster center
        idx = np.nanargmin(difference, 1)
    else:
        idx = np.argmin(difference, 1)  # Get the minimum index in u

    u[np.arange(len(u)), idx] = 1  # Set the cluster to 1
    return u


def get_beta(cluster_n, nodes_n, eta, new_alpha, alpha, u):
    """
    :param cluster_n:C, the number of clusters
    :param nodes_n:the number of nodes in total
    :param eta:min(1,1/t^([d/2-1]))
    :param new_alpha:the list of alpha_t+1
    :param alpha:the list of alpha_t
    :param u:the data point i belongs to k-th cluster, equals to z
    :return:beta_t+1
    """
    # The first part of min(B_t+1)
    b_1 = 0
    for k in range(cluster_n):
        b_1 += np.exp(-eta * nodes_n * abs(new_alpha[k] - alpha[k]))
    b_1 /= cluster_n

    # The second part of min(B_t+1)
    b_2 = (1 - np.max(np.sum(u, 0) / nodes_n)) / (np.sum(alpha * np.log(alpha)) * -np.max(alpha))
    new_beta = min(b_1, b_2)

    return new_beta


def Update_cluster(new_alpha, nodes, nodes_n, u, cluster_n, cluster_cen, nodes_dim):
    """
    Update both cluster number and clusters, and update alpha and z at the same time
    :param new_alpha:the list of alpha_t+1
    :param nodes:Data of point(x_i)
    :param nodes_n:the number of nodes in total
    :param u:the data point i belongs to k-th cluster, equals to z
    :param cluster_n:C, the number of clusters
    :param cluster_cen:a_k, the cluster center k
    :param nodes_dim:dimension of node's data
    :return: alpha_t+1, z_t+1, c_t+1, a_t+1, end
    """
    end = False  # If we reach the end of algorithm
    new_cluster_cen = np.array([])
    index = np.where(new_alpha <= (1 / nodes_n))  # smaller than 1/n, considered to be illegitimate proportions
    adj_alpha = new_alpha
    adj_alpha = np.delete(adj_alpha, index)
    adj_alpha = adj_alpha / np.sum(adj_alpha)  # Re-normalize
    new_alpha = adj_alpha
    '''
    np.size return the elements of a array
    For a three-dimension array, the option 3 means the row
    '''

    if np.size(new_alpha, 0) == 1:  # new alpha only have one row, means only one cluster
        end = True
        return new_alpha, u, cluster_n, cluster_cen, end

    new_cluster_n = np.size(new_alpha, 0)  # Get new cluster number
    adj_u = u  # Summon new z
    adj_u = np.delete(adj_u, index, 1)
    for i in range(len(adj_u)):
        if np.sum(adj_u[i])>0:
            adj_u[i] = adj_u[i] / np.sum(adj_u[i])  # Re-normalize
        else:
            adj_u[i] = np.zeros(adj_u.shape[1])  # Test element-wise for NaN and return result as a boolean array.
    new_u = adj_u

    new_cluster_cen = np.empty((new_cluster_n, 2))
    for k in range(new_cluster_n):
        sum_zx = [0, 0]
        sum_z = 0
        for i in range(nodes_n):
            sum_zx += new_u[i, k] * nodes[i]
            sum_z += new_u[i, k]
        if sum_z > 0:
            new_cluster_cen[k] = sum_zx / sum_z
            #new_cluster_cen = np.append(new_cluster_cen, sum_zx / sum_z)
        else:
            new_cluster_cen[k] = np.mean(nodes)
            #new_cluster_cen = np.append(new_cluster_cen, np.mean(nodes))  # remove abandoned cluster
    return new_alpha, new_u, new_cluster_n, new_cluster_cen, end


def converge(new_cluster_n, new_cluster_cen, cluster_cen):
    """
    :param new_cluster_n: c_t+1
    :param new_cluster_cen: a_t+1
    :param cluster_cen: a_t
    :return: the max error for a
    """
    error = np.array([])
    for k in range(new_cluster_n):
        error = np.append(error, np.linalg.norm(new_cluster_cen[k] - cluster_cen[k]))
    err = max(error)

    return err


def UK_mean_origin(nodes, nodes_dim=2, max_cluster=10):
    """
    :param nodes:data of nodes
    :param nodes_dim:dimension of nodes
    :param max_cluster:the maximum cluster
    :return:cluster_n: number of cluster, cluster_cen: center of each cluster,
            cluster: each point belongs to which cluster
    """
    nodes = np.array(nodes)
    thres = 0.1  # Fixed converge
    beta = 1
    gamma = 1
    rate = 0  # The time t for generation
    shift = np.array([[0.0001, 0.0001]])
    # cluster_n = max_cluster
    cluster_n = len(nodes)
    nodes_n = len(nodes)
    # clust_cen = nodes + 0.0001
    cluster_cen = nodes + shift
    alpha = np.ones(cluster_n) * 1 / cluster_n
    err = 10
    t_max = 100
    c_history = []
    u = np.zeros((nodes_n, cluster_n))
    while cluster_n > 1 and err >= thres:
        rate = rate + 1
        u = get_u(nodes, nodes_n, cluster_cen, cluster_n, gamma, alpha, rate)
        gamma = np.exp(-cluster_n / 450)
        # Update alpha
        new_alpha = np.sum(u, 0) / nodes_n + beta / gamma * alpha * (np.log(alpha) - np.sum(alpha * np.log(alpha)))
        a = 1 / rate  # Used to calculate eta
        eta = min(1, a ** np.floor(nodes_dim / 2 - 1))
        new_beta = get_beta(cluster_n, nodes_n, eta, new_alpha, alpha, u)  # Update beta

        new_alpha, new_u, new_cluster_n, new_cluster_cen, end = Update_cluster(new_alpha, nodes, nodes_n, u, cluster_n,
                                                                               cluster_cen, nodes_dim)
        if end:
            break
        if rate >= 600 and new_cluster_n - cluster_n == 0:
            new_beta = 0

        # Update parameters for next generation
        err = converge(new_cluster_n, new_cluster_cen, cluster_cen)
        cluster_cen = new_cluster_cen
        cluster_n = new_cluster_n
        alpha = new_alpha
        beta = new_beta
        u = new_u
        c_history = np.append(c_history, cluster_n)
        print(rate)
        print(cluster_cen)
        print(err)

    cluster = np.array([])
    for i in range(nodes_n):
        index = np.argmax(u[i, :])
        cluster = np.append(cluster, index)
    return cluster_n, cluster_cen

def UK_mean(agents, beta_param, nodes_dim=2, max_cluster=10):
    """
    :param nodes:data of nodes
    :param nodes_dim:dimension of nodes
    :param max_cluster:the maximum cluster
    :return:cluster_n: number of cluster, cluster_cen: center of each cluster,
            cluster: each point belongs to which cluster
    """
    agents_n = len(agents)
    nodes = []
    for agent in agents:
        nodei = [agent.latency[0], agent.Acc[0]]
        nodes.append(nodei)
    nodes = np.array(nodes)
    #print("Ukmean nodes")
    #print(nodes)
    thres = 0.1  # Fixed converge
    beta = beta_param

    #print("Ukmean beta")
    #print(beta)
    gamma = 1
    rate = 0  # The time t for generation
    shift = np.array([[0.0001, 0.0001]])
    # cluster_n = max_cluster
    cluster_n = len(nodes)
    nodes_n = len(nodes)
    # clust_cen = nodes + 0.0001
    cluster_cen = nodes + shift
    alpha = np.ones(cluster_n) * 1 / cluster_n
    err = 10
    t_max = 100
    c_history = []
    u = np.zeros((nodes_n, cluster_n))
    while cluster_n > 1 and err >= thres:
        rate = rate + 1
        u = get_u(nodes, nodes_n, cluster_cen, cluster_n, gamma, alpha, rate)
        gamma = np.exp(-cluster_n / 450)
        # Update alpha
        new_alpha = np.sum(u, 0) / nodes_n + beta / gamma * alpha * (np.log(alpha) - np.sum(alpha * np.log(alpha)))
        a = 1 / rate  # Used to calculate eta
        eta = min(1, a ** np.floor(nodes_dim / 2 - 1))
        new_beta = get_beta(cluster_n, nodes_n, eta, new_alpha, alpha, u)  # Update beta

        new_alpha, new_u, new_cluster_n, new_cluster_cen, end = Update_cluster(new_alpha, nodes, nodes_n, u, cluster_n,
                                                                               cluster_cen, nodes_dim)
        if end:
            break
        if rate >= 600 and new_cluster_n - cluster_n == 0:
            new_beta = 0

        # Update parameters for next generation
        err = converge(new_cluster_n, new_cluster_cen, cluster_cen)
        cluster_cen = new_cluster_cen
        cluster_n = new_cluster_n
        alpha = new_alpha
        beta = beta_param * new_beta
        u = new_u
        c_history = np.append(c_history, cluster_n)

    cluster = []
    cluster_center = []
    cluster_ids = np.zeros(len(nodes))
    for i in range(cluster_n):
        clusteri = []
        cluster_centeri = []
        for j in range(nodes_n):
            index = np.argmax(u[j, :])
            if index == i:
                clusteri.append(agents[j])
                cluster_centeri.append(nodes[j])
                cluster_ids[j] = i
        cluster_center.append(np.sum(cluster_centeri, 0) / len(cluster_centeri))
        cluster.append(clusteri)


    #df = pd.DataFrame(np.c_[nodes ,cluster_ids],columns = ['feature1','feature2','cluster_id'])
    #df['cluster_id'] = df['cluster_id'].astype('i2')
    #df.plot.scatter('feature1','feature2', s = 100,
    #c = list(df['cluster_id']),cmap = 'rainbow',colorbar = False,
    #alpha = 0.6,title = 'sklearn ukmeans cluster result')
    #print(cluster_ids)
    return cluster, cluster_n, cluster_center

def UK_meanlie(agents, beta_param, nodes_dim=2, max_cluster=10):
    """
    :param nodes:data of nodes
    :param nodes_dim:dimension of nodes
    :param max_cluster:the maximum cluster
    :return:cluster_n: number of cluster, cluster_cen: center of each cluster,
            cluster: each point belongs to which cluster
    """
    agents_n = len(agents)
    nodes = []
    for agent in agents:
        if agent.lie == 1:
            nodei = [agent.latency_lie[0], agent.Acc_lie[0]]
        else:
            nodei = [agent.latency[0], agent.Acc[0]]
        nodes.append(nodei)

    nodes = np.array(nodes)
    #print("Ukmean nodes")
    #print(nodes)
    thres = 0.1  # Fixed converge
    beta = beta_param
    gamma = 1
    rate = 0  # The time t for generation
    shift = np.array([[0.0001, 0.0001]])
    # cluster_n = max_cluster
    cluster_n = len(nodes)
    nodes_n = len(nodes)
    # clust_cen = nodes + 0.0001
    cluster_cen = nodes + shift
    alpha = np.ones(cluster_n) * 1 / cluster_n
    err = 10
    t_max = 100
    c_history = []
    u = np.zeros((nodes_n, cluster_n))
    while cluster_n > 1 and err >= thres:
        rate = rate + 1
        u = get_u(nodes, nodes_n, cluster_cen, cluster_n, gamma, alpha, rate)
        gamma = np.exp(-cluster_n / 450)
        # Update alpha
        new_alpha = np.sum(u, 0) / nodes_n + beta / gamma * alpha * (np.log(alpha) - np.sum(alpha * np.log(alpha)))
        a = 1 / rate  # Used to calculate eta
        eta = min(1, a ** np.floor(nodes_dim / 2 - 1))
        new_beta = get_beta(cluster_n, nodes_n, eta, new_alpha, alpha, u)  # Update beta

        new_alpha, new_u, new_cluster_n, new_cluster_cen, end = Update_cluster(new_alpha, nodes, nodes_n, u, cluster_n,
                                                                               cluster_cen, nodes_dim)
        if end:
            break
        if rate >= 600 and new_cluster_n - cluster_n == 0:
            new_beta = 0

        # Update parameters for next generation
        err = converge(new_cluster_n, new_cluster_cen, cluster_cen)
        cluster_cen = new_cluster_cen
        cluster_n = new_cluster_n
        alpha = new_alpha
        beta = beta_param * new_beta
        u = new_u
        c_history = np.append(c_history, cluster_n)

    cluster = []
    for i in range(cluster_n):
        clusteri = []
        for j in range(nodes_n):
            index = np.argmax(u[j, :])
            if index == i:
                clusteri.append(agents[j])
        cluster.append(clusteri)
    return cluster, cluster_n