import numpy as np
import random
from VCG_implement import *
import matplotlib.pyplot as plt
import networkx as nx
import sys, time, json, os, pdb
from numbers import Number
import pandas as pd
import argparse
import ENV

parser = argparse.ArgumentParser()
parser.add_argument('-RC', '--Random_config', type=str, default='./config.json',
                    help='TradeFL random configuration file.')
parser.add_argument('-DC', '--Default_config', type=str, default='./default.json',
                    help='TradeFL default configuration file.')
parser.add_argument('-anum', '--agent_num', type=int, default=10,
                    help='The number of agents.')
parser.add_argument('-de', '--BY_DEFAULT', type=bool, default=True,
                    help='Whether to use default parameters.')
parser.add_argument('-k', '--k', type=float, default=6.074e-27,
                    help='The energy coefficient.')
parser.add_argument('-g', '--G', type=int, default=100,  # args.G
                    help='The accuracy upper bound.')
parser.add_argument('-ss', '--S_step', type=int, default=5,
                    help="The discrete step of S.")
parser.add_argument('-fs', "--F_step", type=int, default=5,
                    help="The discrete step of F.")
parser.add_argument('-step', '--Max_step', type=int, default=2e5,
                    help="The maximum steps.")
parser.add_argument('-theta', '--theta', type=float, default=0.5,
                    help='The gradient coefficient.')
parser.add_argument('-refine', '--refine_number', type=int, default=0,
                    help="The refinement steps.")

args = parser.parse_args()
# if args.BY_DEFAULT:
#     MASL_config = config.Config(args.Default_config)
# else:
#     MASL_config = config.Config(args.Random _config)
with open('default.json') as f:
    VCG_config = json.load(f)

if __name__ == '__main__':

    center_node_list = [[0, 0], [5, 5], [10, 10]]

    CrowdFunding_ENV = ENV.Crowdfunding_env(args.num_per_cluster, center_node_list, args.framework_list)

    exp_step = 0
    auction_epoch = 0

    while True:
        model, utility, c_tax, s_tax = CrowdFunding_ENV.crowdfunding()
        CrowdFunding_ENV.Evolve()
        CrowdFunding_ENV.reset()
        exp_step += 1
        if exp_step > 20:
            break

