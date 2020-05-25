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
#from slot import Slot


class Agent:

    def __init__(self, index, n_agents):
        self._index = index
        self.num_agents = n_agents
        self._f = (self.num_agents - 1) // 3
        self.observations_recieved = None
        self._dont_send = False
        self.value_to_send = None
        self._next_propose_slot_no = 0
        self.prepare_slots = {}    # keeps track of who has proposed which slot; (key:slot #, value:list of agents)
        self.commit_slots = {}     # keeps track of who has committed which slot
        self.prepare_sent = {}     # keeps track of which prepare messages have been already sent
        self.commit_sent = {}      # keeps track of which commit messages have been already sent
        self.permanent_record = {} # permanent record of chose observatin
        self._sent = False

    def get_obs(self):
        """
        Client function
        :return:
        """
        return np.random.randint(65, 75)
      

    async def send_obs_request(self):
        """
        Client function (sends requests)
        :return:
        """
        # Initialize session
        timeout = aiohttp.ClientTimeout(1)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._closed = False

        # Get Observations
        data = self.get_obs()
        await asyncio.sleep(random())
        json_data = {
            'id': (self._index, data),
            'timestamp': time.time(),
            'data': str(data)
        }
        # TODO: Implement rotating view change
        current_leader = 0
        self._got_response = asyncio.Event()

        # Try and send observation
        while 1:
            try:
                await self._session.post(make_url(30000 + current_leader), json=json_data)
                await asyncio.wait_for(self._got_response.wait(), 5)
            except:
                pass
            else:
                if not self._closed:
                    await self._session.close()
                    print("CLOSED", self._index)
                    self._closed = True
                    break

    async def close(self, agents):
        """
        Debugging function to close all connections. Called by leader.
        :return:
        """
        for i in agents:
            print("closing : ", i)
            while 1:
                try:
                    await self._session.post(make_url(30000 + i, endpoint='get_reply'), json='{}')
                    await asyncio.wait_for(self._got_response.wait(), 5)
                except:
                    pass
                else:
                    is_sent = True
                    break
            print("closed {}".format(i))

    async def post(self, agents, endpoint, json_data):
        '''
        Broadcast json_data to all node in nodes with given command.
        input:
            agents: list of agents
            endpoint: destination of message
            json_data: Data in json format.
        '''
        if not self._session:
            print("not self._session")
            timeout = aiohttp.ClientTimeout(1)
            self._session = aiohttp.ClientSession(timeout=timeout)
        for i in agents:
            try:
                _ = await self._session.post(
                    make_url(30000+i, endpoint), json=json_data)
            except Exception as e:
                pass


    async def get_obs_request(self, get_obs_request):
        """
        Node function. If not leader, re-directs to leader.
        Calls preprepare function.
        :return:
        """
        if self._index != 0:  # if self is not leader redirect to leader
            raise web.HTTPTemporaryRedirect(self.make_url(0, 'get_obs_request'))
        else:
            j = await get_obs_request.json()
            if self.observations_recieved is None:
                self.observations_recieved = {}
                self.time_started = time.time()
            self.observations_recieved[j['id'][0]] = j['data']

            if not self._sent: print(self.observations_recieved)
            if (self.value_to_send is None) and (len(self.observations_recieved.keys()) > ((2 * self._f) + 1)):
                vals = (list(self.observations_recieved.values()))
                self.value_to_send = np.median(list(map(int, vals)))
                #### TODO ^ REPLACE ABOVE WITH BETTER SCHEME BASED ON OBSERVATION DISTRIBUTION

            if self.value_to_send is not None and not self._sent:
                self._sent = True
                print("VALUE TO SEND:", self.value_to_send)
                request = {
                    'leader': self._index,
                    'data': self.value_to_send}
                await self.preprepare(request)
                return web.Response()

    async def preprepare(self, request):
        """
        Sends pre-prepare messages to all agents. Only the leader node executes this.

        json_data: Json-transformed web request from client
                {
                    index: client id,
                    data: "int"
                }
        :return:
        """

        # create slot object
        this_slot_no = str(self._next_propose_slot_no)
        # increment slot number
        self._next_propose_slot_no = int(this_slot_no) + 1

        print("Agent {} on preprepare, propose at slot {}".format(self._index, int(this_slot_no)))
        preprepare_msg = {
            'leader': self._index,
            'proposal': {
                this_slot_no: request
            },
            'type': 'preprepare'
        }

        await self.post(np.arange(self.num_agents), "prepare", preprepare_msg)

    async def prepare(self, preprepare_msg):
        """
        Takes pre-prepare message and if it looks good sends prepare messages to all agents
        :return:
        """
        preprepare_msg = await preprepare_msg.json()

        #########################
        # If data is very different from own data, then ask for leader change ###
        # TODO
        #########################


        # If data seems valid, send prepare message to all agents
        for slot_no, data in preprepare_msg['proposal'].items():
            prepare_msg = {
                'index': self._index,
                'proposal': {
                    slot_no: data
                },
                'type': 'prepare'
            }
            print("Agent {} sent prepare".format(self._index))
            await self.post(np.arange(self.num_agents), 'commit', prepare_msg)

        return web.Response()

    async def commit(self, prepare_msg):
        """
        After getting more than 2f+1 prepare messages, send commit message to all agents
        :return:
        """
        prepare_msg = await prepare_msg.json()
        assert(prepare_msg['type'] == 'prepare')

        for slot_no, data in prepare_msg['proposal'].items():
            if slot_no not in self.prepare_slots.keys():
                self.prepare_slots[slot_no] = []
            self.prepare_slots[slot_no].append(prepare_msg['index'])

            if (len(self.prepare_slots[slot_no]) > 2*self._f + 1) and (slot_no not in self.prepare_sent.keys()):
                self.prepare_sent[slot_no] = True
                commit_msg = {
                    'index': self._index,
                    'proposal': {
                        slot_no: data
                    },
                    'type': 'commit'
                }
                print("Agent {} sent commit".format(self._index))
                await self.post(np.arange(self.num_agents), 'reply', commit_msg)

    async def reply(self, commit_msg):
        """
        After getting more than 2f+1 commit messages, saves commit certificate,
        save commit observation
        :return:
        """
        commit_msg = await commit_msg.json()
        assert(commit_msg['type'] == 'commit')
        
        for slot_no, data in commit_msg['proposal'].items():
            if slot_no not in self.commit_slots.keys():
                self.commit_slots[slot_no] = []
            self.commit_slots[slot_no].append(commit_msg['index'])
            
            if (len(self.commit_slots[slot_no]) > 2*self._f + 1)  and (slot_no not in self.commit_sent.keys()):
                print("Agent {} committed".format(self._index), data)
                self.commit_sent[slot_no] = True
                self.permanent_record[slot_no] = data
                try:
                    self._got_response.set()
                except:
                    pass
                  
                if not self._closed:
                    await self._session.close()
                    print("CLOSED", self._index)
                    self._closed = True

                    
def make_url(node, endpoint='get_obs_request'):
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
#         web.post('/get_reply', agent.get_reply),
        web.post('/preprepare', agent.preprepare),
        web.post('/prepare', agent.prepare),
        web.post('/commit', agent.commit),
        web.post('/reply', agent.reply),
    ])

    web.run_app(app, host="localhost", port=port, access_log=None)


if __name__ == "__main__":
    main()