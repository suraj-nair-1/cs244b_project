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
import subprocess
from subprocess import Popen


class TempEnv():
    def __init__(self):
        self.true_state = 70

    def reset(self):
        self.true_state = np.random.randint(65, 75)
        return self.true_state


def make_url(node, endpoint=None):
    return "http://{}:{}/{}".format("localhost", node, endpoint)

async def get_response(session, url, data):
    try:
        result = await session.get(url, json=data)
        js = await result.json()
        return(js['res'])
    except:
        print("DONE!")
  
async def send_msg(session, url, data):
    try:
        result = await session.post(url, json=data)
    except:
        print("DONE!")


def main():
    parser = argparse.ArgumentParser(description='PBFT Node')
    parser.add_argument('-n', '--num_agents', type=int, help='node index')
    parser.add_argument('--num_eps', type=int, help='node index')
    parser.add_argument('--num_faulty', type=int, help='node index')
    #parser.add_argument('-env', '--env', type=str, help='domain')
    args = parser.parse_args()

    env = TempEnv()

    subprocess.run("rm ./logs/agent*", shell=True)
    print("DELETED OLD FILES")

    for i in range(args.num_agents):
        # subprocess.run(f"fuser -k {30000+i}/tcp", shell=True)
        subprocess.run(f"lsof -nti:{30000+i} | xargs kill -9", shell=True)

    print("KILLED OLD PROC")
    p = []
    for i in range(args.num_agents):
        p.append(Popen(f"python ./run_agent.py -i {i} -n {args.num_agents} -f {args.num_faulty}", shell=True))
    print("STARTED NEW PROC")
    time.sleep(10)

    timeout = aiohttp.ClientTimeout(1000)
    sess = aiohttp.ClientSession(timeout=timeout)

    true_obs = []
    for ep in range(args.num_eps):
        print("EPISODE:", ep)
        obs = env.reset()
        true_obs.append([obs])
        loop = asyncio.get_event_loop()
        futures = [send_msg(sess, make_url(30000 + i, "setobs"), {"obs": obs}) for i in range(args.num_agents)]
        loop.run_until_complete(asyncio.gather(*futures))

        loop = asyncio.get_event_loop()
        futures = [send_msg(sess, make_url(30000 + i, "send_obs_request"), {}) for i in range(args.num_agents)]
        loop.run_until_complete(asyncio.gather(*futures))

        loop = asyncio.get_event_loop()
        futures = [send_msg(sess, make_url(30000 + i, "reopen"), {}) for i in range(args.num_agents)]
        loop.run_until_complete(asyncio.gather(*futures))
        
    loop = asyncio.get_event_loop()
    futures = [get_response(sess, make_url(30000 + i, "get_results"), {}) for i in range(args.num_agents)]
    rs = loop.run_until_complete(asyncio.gather(*futures))
    results = np.array(rs)
    true_obs = np.array(true_obs)
    print(true_obs.shape)
    true_obs = true_obs.repeat(args.num_agents, 1)
    print(true_obs.shape)
    results = results.swapaxes(0, 1)
    print(results.shape)
    results = np.stack([results, true_obs])
    print(results.shape)
    np.save(f"logs/results_nagents_{args.num_agents}_numeps_{args.num_eps}_nfaulty_{args.num_faulty}_type_random.npy", results)
        
    asyncio.ensure_future(sess.close())
    time.sleep(3)
    for pr in p:
        pr.terminate()
    print("KILLED NEW PROC")


if __name__ == "__main__":
    main()
