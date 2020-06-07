from random import random, randint
from collections import Counter
import json
import sys
import argparse
import time
import asyncio
import aiohttp
from aiohttp import web
import numpy as np
from agent import Agent
from faulty_agents import *


def create_agent(args, is_byzantine=False):
    if not is_byzantine:
        return Agent(args.index, args.num_agents)
    else:

        #if args.num_faulty >= (args.num_agents-1)//3:  # if args.num_faulty is invalid, just use maximum # faulty agents
        #    args.num_faulty = ((args.num_agents-1)//3) - 1
        if args.index < args.num_faulty:
            print("faulty index", args.index)
            #FaultyAgent = np.random.choice(faulty_agents_list, 1)[0]   # randomly picks a faulty agent
            FaultyAgent = faulty_agents_list[args.index]                # picks faulty agent in order
            agent = FaultyAgent(args.index, args.num_agents)
        else:
            print("regular index", args.index)
            agent = Agent(args.index, args.num_agents)
        return agent


def main():
    parser = argparse.ArgumentParser(description='PBFT Node')
    parser.add_argument('-i', '--index', type=int, help='node index')
    parser.add_argument('-n', '--num_agents', type=int, help='node index')
    parser.add_argument('-f', '--num_faulty', type=int, help='must be less than (num_agents-1)//3')
    #parser.add_argument('-ft', '--faulty_type', type=str, help='type of faulty agent')
    parser.add_argument('--control', action='store_true')
    args = parser.parse_args()

    print("STARTING", args)
    agent = create_agent(args, is_byzantine=False)
    port = 30000 + args.index

    time.sleep(np.random.randint(2))

    app = web.Application()
    app.add_routes([
        web.post('/send_obs_request', agent.send_obs_request),
        web.post('/get_obs_request', agent.get_obs_request),
        web.post('/preprepare', agent.preprepare),
        web.post('/prepare', agent.prepare),
        web.post('/commit', agent.commit),
        web.post('/reply', agent.reply),
        web.post('/reopen', agent.reopen),
        web.post('/setobs', agent.setobs),
        web.post('/leader_change', agent.leader_change),
    ])

    web.run_app(app, host="localhost", port=port, access_log=None)


if __name__ == "__main__":
    main()
