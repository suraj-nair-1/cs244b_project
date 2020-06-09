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
import multiagent_gridworld
import gym
import ast
import pickle


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

def run_experiment(env, num_agents, num_faulty, num_obs, sample_no, method):

    subprocess.run("rm ./logs/agent/agent*", shell=True)
    print("DELETED OLD FILES")

    for i in range(num_agents):
        #subprocess.run(f"fuser -k {30000+i}/tcp", shell=True)
        subprocess.run(f"sudo lsof -i tcp:30000+i ", shell=True)
#         subprocess.run(f"lsof -nti:{30000+i} | xargs kill -9", shell=True)

    print("KILLED OLD PROC")
    p = []
    for i in range(num_agents):
        p.append(Popen(f"python ./run_agent.py -i {i} -n {num_agents} -o {num_obs} -f {num_faulty} -m {method}", shell=True))
    print("STARTED NEW PROC")
    time.sleep(10)
    
    timeout = aiohttp.ClientTimeout(1000)
    sess = aiohttp.ClientSession(timeout=timeout)

    true_obs = []
    noisy_obs = []

    grid, obstacles = env.reset()
    obstacles_str = json.dumps(obstacles)
    done = False
    step = 0
    while not done:
        step += 1
        print("STEP:", step)
        #print("obs: ",obstacles)
        true_obs.append(obstacles)

        loop = asyncio.get_event_loop()
        futures = [send_msg(sess, make_url(30000 + i, "setobs"), {"obs": obstacles_str}) for i in range(num_agents)]
        loop.run_until_complete(asyncio.gather(*futures))

        loop = asyncio.get_event_loop()
        futures = [send_msg(sess, make_url(30000 + i, "send_obs_request"), {}) for i in range(num_agents)]
        loop.run_until_complete(asyncio.gather(*futures))

        loop = asyncio.get_event_loop()
        futures = [send_msg(sess, make_url(30000 + i, "reopen"), {}) for i in range(num_agents)]
        loop.run_until_complete(asyncio.gather(*futures))

        loop = asyncio.get_event_loop()
        futures = [get_response(sess, make_url(30000 + i, "get_results"), {}) for i in range(num_agents)]
        rs = loop.run_until_complete(asyncio.gather(*futures))
        #print("RS: ", rs)
        rs = [ast.literal_eval(x) for x in rs]
        results = np.array(rs[0]).astype(np.int32)
        #print(results)
        #print(results.shape)  #  num_obstacles, 2
        noisy_obs.append(results)

        _, obstacles, done, _ = env.step(results)
        obstacles_str = json.dumps(obstacles)

        if step >= 300:
            break


    true_obs = np.array(true_obs)
    noisy_obs = np.array(noisy_obs)
    #print(true_obs.shape, noisy_obs.shape)  # num_steps, num_obstacles, num_agents
    results = np.stack([noisy_obs, true_obs])
    #print(results.shape)
    results_dict = {'steps': step, 'error':results}
    output_dir = open(f"logs/control_results/results_nagents_{num_agents}_nfaulty_{num_faulty}_nobs_{num_obs}_method_{method}_sample_{sample_no}.pkl", 'wb')
    pickle.dump(results_dict, output_dir)
    output_dir.close()
    #loop = asyncio.get_event_loop()
    #futures = [get_response(sess, make_url(30000 + i, "get_results"), {}) for i in range(num_agents)]
    #rs = loop.run_until_complete(asyncio.gather(*futures))
    #results = np.array(rs)
    #print(results.shape)
    #true_obs = np.array(true_obs)
    #true_obs = true_obs.repeat(num_agents, 1)
    #results = results.swapaxes(0, 1)
    #results = np.stack([results, true_obs])
    #np.save(f"logs/results/results_nagents_{num_agents}_numeps_{num_eps}_nfaulty_{num_faulty}_faultytype_randomobs_method_{method}.npy", results)
        
    asyncio.ensure_future(sess.close())
    time.sleep(3)
    for pr in p:
        pr.terminate()
    print("KILLED NEW PROC")
    print("TOTAL STEPS: ", step)
        

def main():
    parser = argparse.ArgumentParser(description='PBFT Node')
    parser.add_argument('-n', '--num_agents', type=int, help='node index')
    parser.add_argument('--num_eps', type=int, help='node index')
    parser.add_argument('-o', '--num_obs', type=int, help='node index')
    parser.add_argument('--num_faulty', type=int, help='node index')
    parser.add_argument('--method', type=str, help='node index')
    #parser.add_argument('-env', '--env', type=str, help='domain')
    args = parser.parse_args()

    num_agents = 4
    num_faulty = 1
    for num_obs in [1, 10, 20, 30, 40, 50]:
        for sample in range(10):
            env = gym.make("MultiGrid-v0")
            env.num_agents = num_agents
            env.num_obstacles = num_obs
            run_experiment(env, num_agents, num_faulty, num_obs, sample, args.method)
        print(f"FINISHED NUMOBS {num_obs} SAMPLE {sample}")



if __name__ == "__main__":
    main()
