import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import Player_class


def ini_agentvalue(agent_cluster, model_list, value=100):
    agent_list = []
    for agent in agent_cluster:
        # agent_list.append(np.random.randint(value, size=model_num))
        value_list = []
        for model in model_list:
            value = agent.get_evaluation(model)
            value_list.append(value)
        agent_list.append(value_list)
    return agent_list


def Winner_determine(model_value, model_cost):
    model_utility = model_value - model_cost
    winner_index = np.argmax(model_utility)

    #if np.max(model_utility) < 0:
    #    model_utility[winner_index] = 0
        #print("No Satisfied Outcome")
    return winner_index, model_utility


def VCG_auction(bidder_list, server_list, excost_list):
    model_value = np.sum(bidder_list, axis=0)
    model_cost = np.sum(server_list, axis=0)
    winner_model, model_utility = Winner_determine(model_value, model_cost)
    bidder_num = len(bidder_list)
    server_num = len(server_list)

    bidder_tax = np.zeros(bidder_num)
    bid_negative_uti = 0
    server_tax = np.zeros(server_num)

    for i in range(bidder_num):
        margin_value = model_value - bidder_list[i]
        margin_winner, margin_utility = Winner_determine(margin_value, model_cost)
        if margin_winner != winner_model:
            bidder_tax[i] = abs(margin_utility[margin_winner] - model_utility[winner_model])
    randompay = []
    for i in range(bidder_num):
        randompay.append(excost_list[i][winner_model] * 1.15)
    while (np.sum(randompay) < model_cost[winner_model]):
        for i in range(bidder_num):
            randompay[i] += 0.01 * model_cost[winner_model]/len(bidder_list)

    reward_relax = model_utility[winner_model][0]
    reward_nonrelax = model_utility[winner_model][0]
    bid_negative_num = 0

    for i in range(bidder_num):
        self_utility = bidder_list[i][winner_model] - (randompay[i] + bidder_tax[i])
        if self_utility < 0:
            bid_negative_uti -= self_utility[0] * 2
            bid_negative_num +=1

    reward_relax -= bid_negative_uti
    reward_nonrelax -= bid_negative_uti
    reward_nonrelax = reward_nonrelax
    if bid_negative_uti > 0 :
        reward_nonrelax = 0

    if model_utility[winner_model] <= 0:
        bid_negative_uti = 0
        model_utility[winner_model] = 0
        reward_nonrelax = 0
        reward_relax = 0
        bid_negative_num = 0
    if reward_relax < 0:
        reward_relax = 0
    if reward_nonrelax < 0:
        reward_nonrelax = 0

    for j in range(server_num):
        margin_cost = model_cost - server_list[j]
        margin_winner, margin_utility = Winner_determine(model_value, margin_cost)
        if margin_winner != winner_model:
            server_tax[j] = abs(margin_utility[margin_winner] - model_utility[winner_model])

    return winner_model, model_utility[winner_model], bidder_tax, server_tax, bid_negative_uti, model_cost[winner_model], model_value[winner_model], reward_relax, reward_nonrelax, bid_negative_num

    
def VCG_auction_lie(bidder_list, server_list, bidder_value_list, excost_list):
    model_value = np.sum(bidder_list, axis=0)
    model_cost = np.sum(server_list, axis=0)
    real_model_value = np.sum(bidder_value_list, axis=0)
    winner_model, model_utility = Winner_determine(model_value, model_cost)
    model_utility_real = real_model_value - model_cost
    bidder_num = len(bidder_list)
    server_num = len(server_list)

    bidder_tax = np.zeros(bidder_num)
    bid_negative_uti = 0
    server_tax = np.zeros(server_num)
    bidder_tax2 = np.zeros(bidder_num)

    for i in range(bidder_num):
        margin_value = model_value - bidder_list[i]
        margin_winner, margin_utility = Winner_determine(margin_value, model_cost)
        if margin_winner != winner_model:
            bidder_tax[i] = abs(margin_utility[margin_winner] - margin_utility[winner_model])
    for i in range(bidder_num):
        margin_value = real_model_value - bidder_value_list[i]
        margin_winner, margin_utility = Winner_determine(margin_value, model_cost)
        if margin_winner != winner_model:
            bidder_tax2[i] = abs(margin_utility[margin_winner] - margin_utility[winner_model])
    randompay = []
    for i in range(bidder_num):
        randompay.append(excost_list[i][winner_model] * 1.15)
    while (np.sum(randompay) < model_cost[winner_model]):
        for i in range(bidder_num):
            randompay[i] += 0.01 * model_cost[winner_model]/len(bidder_list)

    bidders_utility = []
    for i in range(bidder_num):
        self_utility = bidder_value_list[i][winner_model] - (bidder_tax[i] + randompay[i])
        bidders_utility.append(bidder_value_list[i][winner_model] - (bidder_tax[i] + randompay[i]))
        if self_utility < 0:
            bid_negative_uti += 1

    if model_utility[winner_model] <= 0:
        bid_negative_uti = 0
        model_utility[winner_model] = 0
        model_utility_real[winner_model] = 0
        bidders_utility = [0]
    if model_utility_real[winner_model] < 0:
        bid_negative_uti = 0
        model_utility_real[winner_model] = 0

    for j in range(server_num):
        margin_cost = model_cost - server_list[j]
        margin_winner, margin_utility = Winner_determine(model_value, margin_cost)
        if margin_winner != winner_model:
            server_tax[j] = abs(margin_utility[margin_winner] - model_utility[winner_model])

    return winner_model, model_utility_real[winner_model], bidder_tax, server_tax, bid_negative_uti, bidders_utility
