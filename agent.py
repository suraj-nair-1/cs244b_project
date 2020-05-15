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

    def __init__(self, index, n_agents):
        self._index = index
        self.num_agents = n_agents
        self._f = (self.num_agents - 1) // 3
        self.observations_recieved = None
        self._dont_send = False

    def get_obs(self):
        """
        Client function
        :return:
        """
        return np.random.randint(65, 75)
      
    async def get_reply(self, request):
        json_data = await request.json()
        print(self._index, "REPLY:",  json_data)
        try:
            self._got_response.set()
        except:
            self._dont_send = True
        return web.Response()

    async def send_obs_request(self):
        """
        Client function (sends requests)
        :return:
        """
        timeout = aiohttp.ClientTimeout(5)
        self._session = aiohttp.ClientSession(timeout = timeout)
        
        is_sent = False
        data = self.get_obs()
        await asyncio.sleep(random())
        json_data = {
                'id': (self._index, data),
                'timestamp': time.time(),
                'data': str(data)        
            }
        current_leader = 0
        self._got_response = asyncio.Event()
        
        if not self._dont_send:
            while 1:
                try:
                    await self._session.post(make_url(30000 + current_leader), json=json_data)
                    await asyncio.wait_for(self._got_response.wait(), 5)
                except:
                    pass
                else:
                    print(self._index, "SENT!")
                    is_sent = True
                if is_sent:
                    break

        await self._session.close()
        print("CLOSED", self._index)

    async def get_obs_request(self, get_obs_request):
        """
        Node function. If not leader, re-directs to leader.
        Calls preprepare function.
        :return:
        """
        j = await get_obs_request.json()
        if self.observations_recieved is None:
            self.observations_recieved = {}
        self.observations_recieved[j['id'][0]] = j['data']
        value_to_send = None
        print(self.observations_recieved)
        if len(self.observations_recieved.keys()) >  ((2 * self._f) +1):
            vals = (list(self.observations_recieved.values()))
            value_to_send = np.median(list(map(int, vals)))
            
            #### TODO ^ REPLACE ABOVE WITH BETTER SCHEME BASED ON OBSERVATION DISTRIBUTION
        
        ### TODO
        # DO PBFT STYLE CONSENSUS ON OBSERVATION
        # GET SOME RESULT
        ###
        if value_to_send is not None:
            reply_msg = {
                  'index': self._index,
                  'data': value_to_send }
            open_agents = list(range(self.num_agents))
            while 1:
                key = np.random.choice(open_agents)
                if (key != self._index) and (key in open_agents):
                    try:
                        await self._session.post(make_url(30000 + key, endpoint='get_reply'), json=reply_msg)
                    except:
                        print(open_agents)
                    else:
                        open_agents.remove(key)
                if len(open_agents) == 1:
                    break
            await self._session.post(make_url(30000 + self._index, endpoint='get_reply'), json=reply_msg)
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

def make_url(node, endpoint = 'get_obs_request'):
    return "http://{}:{}/{}".format("localhost", node, endpoint)
  
def main():
    parser = argparse.ArgumentParser(description='PBFT Node')
    parser.add_argument('-i', '--index', type=int, help='node index')
    parser.add_argument('-n', '--num_agents', type=int, help='node index')
    args = parser.parse_args()
    
    print("STARTING", args)
    agent = Agent(args.index, args.num_agents)
    port = 30000 + args.index

#     time.sleep(np.random.randint(10))
    asyncio.ensure_future(agent.send_obs_request())

    app = web.Application()
    app.add_routes([
        web.post('/get_obs_request', agent.get_obs_request),
        web.post('/get_reply', agent.get_reply)
        ])

    
    web.run_app(app, host="localhost", port=port, access_log=None)


if __name__ == "__main__":
    main()