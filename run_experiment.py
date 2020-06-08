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

def run_experiment(env, num_agents, num_faulty, num_eps, method, envtype):
    
    subprocess.run("rm ./logs/agent/agent*", shell=True)
    print("DELETED OLD FILES")

    for i in range(num_agents):
        subprocess.run(f"fuser -k {30000+i}/tcp", shell=True)
#         subprocess.run(f"lsof -nti:{30000+i} | xargs kill -9", shell=True)

    print("KILLED OLD PROC")
  
    if envtype == 'miniworld':
        addon = 'xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" '
    else:
        addon = ''
    p = []
    for i in range(num_agents):
        p.append(Popen(f"{addon}python ./run_agent.py -i {i} -n {num_agents} -f {num_faulty} -m {method} -e {envtype}", shell=True))
    print("STARTED NEW PROC")
    time.sleep(20)
    
    timeout = aiohttp.ClientTimeout(1000)
    sess = aiohttp.ClientSession(timeout=timeout)

    true_obs = []
    for ep in range(num_eps):
        print("EPISODE:", ep)
        obs = env.reset()
        true_obs.append([obs])
        if envtype == 'temp':
            loop = asyncio.get_event_loop()
            futures = [send_msg(sess, make_url(30000 + i, "setobs"), {"obs": obs}) for i in range(num_agents)]
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
    results = []
    for r in rs[:1]:
        rr = []
        for i in range(len(r)):
            rr.append(list(map(int, r[i].split(','))))
        rr = np.stack(rr)
        results.append(rr)
    results = np.stack(results)
    print(results.shape)
#     print(results)
#     print(results.shape)
#     true_obs = np.array(true_obs)
#     true_obs = true_obs.repeat(num_agents, 1)
#     results = results.swapaxes(0, 1)
#     results = np.stack([results, true_obs])
    np.save(f"logs/results/{envtype}_results_nagents_{num_agents}_numeps_{num_eps}_nfaulty_{num_faulty}_faultytype_randomobs_method_{method}.npy", results)
        
    asyncio.ensure_future(sess.close())
    time.sleep(3)
    for pr in p:
        pr.terminate()
    print("KILLED NEW PROC")
        

def main():
    parser = argparse.ArgumentParser(description='PBFT Node')
    parser.add_argument('-n', '--num_agents', type=int, help='node index')
    parser.add_argument('--num_eps', type=int, help='node index')
    parser.add_argument('--num_faulty', type=int, help='node index')
    parser.add_argument('--method', type=str, help='node index')
    parser.add_argument('--env', type=str, default='temp', help='node index')
    args = parser.parse_args()

    env = TempEnv()

    for f in range(0, args.num_faulty+1):
        run_experiment(env, args.num_agents, f, args.num_eps, args.method, args.env)
        print("DID FAULTY", f)
        time.sleep(10)
    


if __name__ == "__main__":
    main()
