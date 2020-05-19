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
        self.value_to_send = None
        self._network_timeout = 10

    def get_obs(self):
        """
        Client function
        :return:
        """
        return np.random.randint(65, 75)
      
    async def get_reply(self, request):
        json_data = await request.json()
        print(self._index, "GOT REPLY:",  json_data)
        try:
            self._got_response.set()
        except:
            self._dont_send = True
        return web.Response()

    async def _post(self, agents, endpoint, json_data):
        '''
        Broadcast json_data to all node in nodes with given command.
        input:
            agents: list of agents
            endpoint: destination of message
            json_data: Data in json format.
        '''
        if not self._session:
            timeout = aiohttp.ClientTimeout(self._network_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        for i in agents:
            if i != self._index:
                _ = await self._session.post(make_url(i, endpoint), json=json_data)  ## QUESTION ABOUT THE 30000, QUESTION ABOUT LIST OF AGENTS

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
        
        while 1:
            try:
                #await self._session.post(make_url(30000 + current_leader), json=json_data)
                await self._session.post(make_url(current_leader), json=json_data)
                await asyncio.wait_for(self._got_response.wait(), 5)
            except:
                pass
            else:
                print(self._index, "SENT!")
                is_sent = True
            if (is_sent) or self._dont_send:
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
            self.time_started = time.time()
        self.observations_recieved[j['id'][0]] = j['data']
        print(self.observations_recieved)
        if (self.value_to_send is None) and (len(self.observations_recieved.keys()) >  ((2 * self._f) +1)):
            vals = (list(self.observations_recieved.values()))
            self.value_to_send = np.median(list(map(int, vals)))
            #### TODO ^ REPLACE ABOVE WITH BETTER SCHEME BASED ON OBSERVATION DISTRIBUTION
        
        
        ### TODO
        # DO PBFT STYLE CONSENSUS ON OBSERVATION
        # GET SOME RESULT
        ###
        if self.value_to_send is not None:
            reply_msg = {
                  'index': self._index,
                  'data': self.value_to_send }
            open_agents = list(self.observations_recieved.keys())
            while 1:
                key = np.random.choice(open_agents)
                if (key != self._index) and (key in open_agents):
                    try:
                        #await self._session.post(make_url(30000 + key, endpoint='get_reply'), json=reply_msg)
                        await self._session.post(make_url(key, endpoint='get_reply'), json=reply_msg)
                    except:
                        print(j['id'][0], key, open_agents)
                    else:
                        open_agents.remove(key)
                        if key in self.observations_recieved.keys():
                            self.observations_recieved.pop(key)
                if len(open_agents) == 1:
                    break
        ### BECAUSE NOT DOING PREPREPARE/PREPARE/COMMIT, LEADER DOESNT KNOW WHEN ALL AGENTS HAVE RESPONSE. 
        ### SO RIGHT NOW THIS IS A HACKY FIX TO MAKE SURE IT CLOSES
        if time.time() - self.time_started > 30:
            #await self._session.post(make_url(30000 + self._index, endpoint='get_reply'), json=reply_msg)
            await self._session.post(make_url(self._index, endpoint='get_reply'), json=reply_msg)
        return web.Response()

    async def preprepare(self, json_data):
        """
        Sends pre-prepare messages to all agents. Only the leader node executes this.

        json_data: Json-transformed web request from client
                {
                    index: client id,
                    data: "int"
                }
        :return:
        """
        # increment sequence number
        this_slot = str(self._next_propose_slot)
        self._next_propose_slot = int(this_slot) + 1

        print("Node {} on preprepare, propose at slot {}".format(self._index, int(this_slot)))

        preprepare_msg = {
            'leader': self._index,
            'slot_no': this_slot,
            'data': json_data,
            'type': 'preprepare'
        }

        agents = np.arange(self.num_agents)
        await self._post(agents, "PREPREPARE", preprepare_msg)

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
    return "http://{}:{}/{}".format("localhost", 30000+node, endpoint)
  
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