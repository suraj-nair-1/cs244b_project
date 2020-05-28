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
        self.obs = None
        self._next_propose_slot_no = 0
        self.prepare_slots = {}  # keeps track of who has proposed which slot; (key:slot #, value:list of agents)
        self.commit_slots = {}  # keeps track of who has committed which slot
        self.leader_change_slots = {}  # keeps track of who has requested a leader change for which slot
        self.prepare_sent = {}  # keeps track of which prepare messages have been already sent
        self.commit_sent = {}  # keeps track of which commit messages have been already sent
        self.permanent_record = {}  # permanent record of chose observation
        self._sent = False
        self.leader = 0  # leader index initialized to 0
        self.reply_slots = {}
        self.epsilon = 10    # threshold for obs being different; requests leader change if above this threshold.

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
        self.obs = data


        await asyncio.sleep(random())
        json_data = {
            'id': (self._index, data),
            'timestamp': time.time(),
            'data': str(data)
        }
        # TODO: Implement rotating view change
        current_leader = self.leader
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
                    make_url(30000 + i, endpoint), json=json_data)
            except Exception as e:
                pass

    async def get_obs_request(self, get_obs_request):
        """
        Node function. If not leader, re-directs to leader.
        Calls preprepare function.
        :return:
        """
        if self._index != self.leader:  # if self is not leader redirect to leader
            raise web.HTTPTemporaryRedirect(make_url(0, 'get_obs_request'))
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

            # testing faulty leader
            self.obs = 1             # the case where only the leader's observation is faulty, but proposed obs is not;
                                     # everyone should still successfully commit the proposed value
            self.value_to_send = self.obs   # the case where leader's observation and proposed observation is faulty.
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

        for slot_no, data in preprepare_msg['proposal'].items():
            if np.abs(self.obs - data['data']) > self.epsilon:
                # request leader change
                print("Agent {} requests leader change! proposed obs: {}, own obs: {}".format(self._index, data['data'], self.obs))
                # If data is very different from own data, then ask for leader change ###
                leader_change_msg = {
                    'index': self._index,
                    'proposal': {
                        slot_no: data
                    },
                    'type': 'leader_change'
                }
                await self.post(np.arange(self.num_agents), 'leader_change', leader_change_msg)
            else:
                # If data seems valid, send prepare message to all agents
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
        After getting at least 2f+1 prepare messages, send commit message to all agents
        :return:
        """
        prepare_msg = await prepare_msg.json()
        assert (prepare_msg['type'] == 'prepare')

        for slot_no, data in prepare_msg['proposal'].items():
            if slot_no not in self.prepare_slots.keys():
                self.prepare_slots[slot_no] = []
            self.prepare_slots[slot_no].append(prepare_msg['index'])

            if (len(self.prepare_slots[slot_no]) >= 2 * self._f + 1) and (slot_no not in self.prepare_sent.keys()):
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
        After getting at least 2f+1 commit messages, saves commit certificate,
        save commit observation
        :return:
        """
        commit_msg = await commit_msg.json()
        assert (commit_msg['type'] == 'commit')

        for slot_no, data in commit_msg['proposal'].items():
            if slot_no not in self.commit_slots.keys():
                self.commit_slots[slot_no] = []
            self.commit_slots[slot_no].append(commit_msg['index'])

            if (len(self.commit_slots[slot_no]) >= 2 * self._f + 1) and (slot_no not in self.commit_sent.keys()):
                print("Agent {} committed".format(self._index), data)
                self.commit_sent[slot_no] = True
                self.permanent_record[slot_no] = data
                # try:
                #    self._got_response.set()
                # except:
                #    pass
                # if not self._closed:
                #    await self._session.close()
                #    print("CLOSED", self._index)
                #    self._closed = True

        # leader change
        try:
            num_sent_commits = 0
            for slot_no, data in commit_msg['proposal'].items():  # count committed slots
                if self.commit_sent[slot_no]:
                    num_sent_commits += 1
            if num_sent_commits == len(
                    commit_msg['proposal'].keys()):  # if every slot has been committed, change leader
                self.leader = (self.leader + 1) % self.num_agents
                print("Agent {} leader changed to {}!".format(self._index, self.leader))
                try:
                    self._got_response.set()
                except:
                    pass
                if not self._closed:
                    await self._session.close()
                    print("CLOSED", self._index)
                    self._closed = True

        except:
            pass

    async def leader_change(self, leader_change_msg):
        """
        Waits for 2f+1 votes before changing leaders.
        :param leader_change_msg:
        :return:
        """
        leader_change_msg = await leader_change_msg.json()
        assert (leader_change_msg['type'] == 'leader_change')

        for slot_no, data in leader_change_msg['proposal'].items():
            assert (data['leader'] == self.leader)
            print("Agent {} on leader change for slot no {}! Curr leader is {}".format(self._index, slot_no, self.leader))
            if slot_no not in self.leader_change_slots.keys():
                self.leader_change_slots[slot_no] = []
            self.leader_change_slots[slot_no].append(leader_change_msg['index'])

            if (len(self.leader_change_slots[slot_no]) >= 2 * self._f + 1):
                self.leader = (self.leader + 1) % self.num_agents
                print("Agent {} leader changed to {}!".format(self._index, self.leader))
                try:
                    self._got_response.set()
                except:
                    pass
                if not self._closed:
                    await self._session.close()
                    print("CLOSED", self._index)
                    self._closed = True


class FaultyAgent1(Agent):
    """
    Doesn't send prepare messages
    """

    def __init__(self, index, n_agents):
        print("Initializing Faulty Agent 1")
        super().__init__(index, n_agents)

    async def prepare(self, preprepare_msg):
        return web.Response()


class FaultyAgent2(Agent):
    """
    Doesn't send commit messages
    """

    def __init__(self, index, n_agents):
        print("Initializing Faulty Agent 1")
        super().__init__(index, n_agents)

    async def commit(self, prepare_msg):
        pass


def make_url(node, endpoint='get_obs_request'):
    return "http://{}:{}/{}".format("localhost", node, endpoint)


def create_agent(args, is_byzantine=False):
    if not is_byzantine:
        return Agent(args.index, args.num_agents)
    else:
        F = int(np.floor((args.num_agents - 1) / 3))
        if args.index < args.num_agents - F:
            print("regular index", args.index)
            agent = Agent(args.index, args.num_agents)
        else:
            print("faulty index", args.index)
            agent = FaultyAgent2(args.index, args.num_agents)
        return agent


def main():
    parser = argparse.ArgumentParser(description='PBFT Node')
    parser.add_argument('-i', '--index', type=int, help='node index')
    parser.add_argument('-n', '--num_agents', type=int, help='node index')
    args = parser.parse_args()

    print("STARTING", args)
    agent = create_agent(args, is_byzantine=False)
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
        web.post('/leader_change', agent.leader_change),
    ])

    web.run_app(app, host="localhost", port=port, access_log=None)


if __name__ == "__main__":
    main()
