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


async def send_msg(session, url, data):
    try:
        await session.post(url, json=data)
    except:
        print("DONE!")


def main():
    parser = argparse.ArgumentParser(description='PBFT Node')
    parser.add_argument('-n', '--num_agents', type=int, help='node index')
    parser.add_argument('--num_eps', type=int, help='node index')
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
        p.append(Popen(f"python ./run_agent.py -i {i} -n {args.num_agents} -f 2", shell=True))
    print("STARTED NEW PROC")
    time.sleep(10)

    timeout = aiohttp.ClientTimeout(1000)
    sess = aiohttp.ClientSession(timeout=timeout)

    for ep in range(args.num_eps):
        print("EPISODE:", ep)
        obs = env.reset()
        loop = asyncio.get_event_loop()
        futures = [send_msg(sess, make_url(30000 + i, "setobs"), {"obs": obs}) for i in range(args.num_agents)]
        loop.run_until_complete(asyncio.gather(*futures))

        loop = asyncio.get_event_loop()
        futures = [send_msg(sess, make_url(30000 + i, "send_obs_request"), {}) for i in range(args.num_agents)]
        loop.run_until_complete(asyncio.gather(*futures))

        loop = asyncio.get_event_loop()
        futures = [send_msg(sess, make_url(30000 + i, "reopen"), {}) for i in range(args.num_agents)]
        loop.run_until_complete(asyncio.gather(*futures))
    #         time.sleep(1)
    sess.close()
    time.sleep(3)
    for pr in p:
        pr.terminate()
    print("KILLED NEW PROC")


if __name__ == "__main__":
    main()
