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

class Agent:

    def __init__(self, index):
        self._index = index
        pass

    def get_obs(self):
        """
        Client function
        :return:
        """
        pass

    async def send_obs_request(self):
        """
        Client function (sends requests)
        :return:
        """
        timeout = aiohttp.ClientTimeout(5)
        self._session = aiohttp.ClientSession(timeout = timeout)
        
        data = np.random.randint(65, 75)
        await asyncio.sleep(random())
        json_data = {
                'id': (self._index, data),
                'timestamp': time.time(),
                'data': str(data)        
            }
        
        print("SENDING", self._index)
        await self._session.post(make_url(30000 + (self._index)), json=json_data)
        print("SENT", self._index)
        await self._session.close()
        print("CLOSED", self._index)

    async def get_obs_request(self, get_obs_request):
        """
        Node function. If not leader, re-directs to leader.
        Calls preprepare function.
        :return:
        """
        j = await get_obs_request.json()
        print(self._index, "GOT REQUEST", j)
        return web.Response()

    def preprepare(self):
        """
        Sends pre-prepare messages to all agents
        :return:
        """
        pass

    def prepare(self):
        """
        Takes pre-prepare message and if it looks good sends prepare messages to all agents
        :return:
        """
        pass

    def commit(self):
        """
        After getting more than 2f+1 prepare messages, send commit message to all agents
        :return:
        """
        pass

    def reply(self):
        """
        After getting more than 2f+1 commit messages, saves commit certificate,
        save commit observation, and send reply back to agent who proposed chosen observation
        :return:
        """
        pass

def make_url(node):
    return "http://{}:{}/{}".format("localhost", node, 'get_obs_request')
  
def main():
    parser = argparse.ArgumentParser(description='PBFT Node')
    parser.add_argument('-i', '--index', type=int, help='node index')
    args = parser.parse_args()
    
    print("STARTING", args)
    agent = Agent(args.index)
    port = 30000 + args.index

    time.sleep(np.random.randint(10))
    asyncio.ensure_future(agent.send_obs_request())

    app = web.Application()
    app.add_routes([
        web.post('/get_obs_request', agent.get_obs_request)
        ])

    
    web.run_app(app, host="localhost", port=port, access_log=None)


if __name__ == "__main__":
    main()