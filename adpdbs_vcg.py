import numpy as np
import ENV
import copy
import Player_class
import os
from VCG_implement import *
import sys, time, json, os, pdb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-WR', '--work_rounds', type=int, default=100,
                    help='The total work rounds.')
parser.add_argument('-FC', '--Framework_config', type=str, default='./config.json',
                    help='Framework for agents and summarize.')
parser.add_argument('-anc', '--agents_center', type=int, default=50,
                    help="The number of agents per center .")
parser.add_argument('-r', "--dbscan_r", type=int, default=5,
                    help="The distance to cluster center for node in dbscan.")
parser.add_argument('-R', '--dbscan_diameter', type=int, default=9,
                    help="The distance to cluster center for node in dbscan.")
parser.add_argument('-rho', '--dbscan_rho', type=int, default=5,
                    help='The gradient coefficient.')
parser.add_argument('-eps', '--cluster_eps', type=int, default=3,
                    help='The gradient coefficient.')
parser.add_argument('-l', '--cluster_learnrate', type=float, default=0.1,
                    help='Learning rate of dbscan cluster.')
args = parser.parse_args()

with open(args.Framework_config, 'r') as f:
    config = json.load(f)

center_node_list = config["center_node_list"]
frameworks = [
    Player_class.Framework(
        properties=frame["properties"],
        cost=frame["cost"],
        index=frame["id"],
        Name=frame["name"]
    )
    for frame in config["frameworks"]
]
CrowdFunding_ENV = ENV.Crowdfunding_env(args.agents_center, center_node_list, frameworks, "line")

agents = CrowdFunding_ENV.Client_list
agents_n = len(agents)
nodes = []
for agent in agents:
    nodei = [agent.latency[0], agent.Acc[0]]
    nodes.append(nodei)

exp_step = 0
exp_step_total = args.work_rounds
dbscan_utility = 0
dbscan_utility_list = [0]
dbscan_cost_list = [0] * (exp_step_total + 1)
dbscan_value_list = [0] * (exp_step_total + 1)
dbscan_reward_list = [0] * (exp_step_total + 1)
dbscan_rewardnon_list = [0] * (exp_step_total + 1)
dbscan_penalty_list = [0] * (exp_step_total + 1)
dbscan_utility_oneround_list = [0] * (exp_step_total + 1)
dbscan_cost_oneround_list = [0] * (exp_step_total + 1)
dbscan_value_oneround_list = [0] * (exp_step_total + 1)
dbscan_reward_oneround_list = [0] * (exp_step_total + 1)
dbscan_rewardnon_oneround_list = [0] * (exp_step_total + 1)
dbscan_penalty_oneround_list = [0] * (exp_step_total + 1)
dbscan_utility_period_list = [0]
dbscan_negative = 0
dbscan_negative_list = []
dbscan_negative_num_list = []
dbscan_negative_nonadd = 0
dbscan_negative_list_nonadd = []
dbscan_negative_list_negative = [0]
dbscan_center = []
dbscan_range = args.cluster_eps
dbscan_maxeps = args.dbscan_r
dbscan_absr = args.dbscan_diameter
dbscan_learn = args.cluster_learnrate
dbscan_point = args.dbscan_rho

while True:
    CrowdFunding_ENV.ser_cluster = [CrowdFunding_ENV.Server_list]
    CrowdFunding_ENV.ser_cluster_num = len(CrowdFunding_ENV.Server_list)
    CrowdFunding_ENV.learn = dbscan_learn
    CrowdFunding_ENV.absmaxr = dbscan_absr
    CrowdFunding_ENV.maxeps = dbscan_maxeps
    CrowdFunding_ENV.cluster_center = dbscan_center
    CrowdFunding_ENV.dbscan_range = dbscan_range
    CrowdFunding_ENV.dbscan_point = dbscan_point


    print("Round: ", exp_step)
    if(exp_step > 90):
        dbscan_learn = 0.4

    CrowdFunding_ENV = copy.deepcopy(CrowdFunding_ENV)
    CrowdFunding_ENV.cli_cluster, CrowdFunding_ENV.cli_cluster_num = CrowdFunding_ENV.cluster(CrowdFunding_ENV.Client_list, "dbscan")
    model, utility, c_tax, s_tax, cl_cost, cl_value, r_relax, r_nonrelax, non_utility, non_num= CrowdFunding_ENV.crowdfunding()
    sumutility = np.sum(utility)
    if sumutility<0:
        sumutility = 0
    #print(utility)
    dbscan_negative_num_list.append(non_num)
    dbscan_utility += sumutility
    dbscan_utility_list.append(dbscan_utility)
    dbscan_cost_list[exp_step+1] += (dbscan_cost_list[exp_step] + cl_cost[0])
    dbscan_value_list[exp_step+1] += (dbscan_value_list[exp_step] + cl_value[0])
    dbscan_penalty_list[exp_step+1] += (dbscan_penalty_list[exp_step] + CrowdFunding_ENV.clientutilityfalse)
    dbscan_utility_oneround_list[exp_step] += sumutility
    dbscan_penalty_oneround_list[exp_step] += CrowdFunding_ENV.clientutilityfalse
    dbscan_reward_oneround_list[exp_step] += r_relax
    dbscan_rewardnon_oneround_list[exp_step] += r_nonrelax
    dbscan_reward_list[exp_step+1] += (dbscan_reward_list[exp_step] + r_relax)
    dbscan_rewardnon_list[exp_step+1] += (dbscan_rewardnon_list[exp_step] + r_nonrelax)
    dbscan_negative += CrowdFunding_ENV.clientutilityfalse
    dbscan_negative_nonadd += CrowdFunding_ENV.clientutilityfalse
    dbscan_negative_list_negative.append(CrowdFunding_ENV.clientutilityfalse)
    dbscan_absr = CrowdFunding_ENV.absmaxr
    dbscan_maxeps = CrowdFunding_ENV.maxeps
    dbscan_center = CrowdFunding_ENV.cluster_center
    dbscan_range = CrowdFunding_ENV.dbscan_range
    dbscan_point = CrowdFunding_ENV.dbscan_point
    dbscan_learn = CrowdFunding_ENV.learn

    exp_step += 1
    if (exp_step >= exp_step_total) :
        break
    CrowdFunding_ENV.Evolve()
    CrowdFunding_ENV.reset()

print("Utility list:", dbscan_utility_list)
