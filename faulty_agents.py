from agent import *


class F_Leader1(Agent):
    """
    Doesn't send observation request
    """

    def __init__(self, index, n_agents, method):
        print("Initializing Faulty Leader 1")
        super().__init__(index, n_agents, method)

    async def get_obs_request(self, get_obs_request):
        if self._index != self.leader:  # if self is not leader redirect to leader
            self.log("Faulty agent {} on get_obs_req. Not leader so redirecting!".format(self._index))
            raise web.HTTPTemporaryRedirect(make_url(self.leader, 'get_obs_request'))
        else:
            self.log("Faulty agent {} on get_obs_req. Is leader, but not sending an obs!".format(self._index))
            pass


class F_Leader2(Agent):
    """
    Sends faulty observation
    TODO: DEFINE FAULTY OBSERVATION BASED ON GROUND TRUTH VALUE
    """

    def __init__(self, index, n_agents, n_obs, method):
        print("Initializing Faulty Leader 2")
        super().__init__(index, n_agents, n_obs, method)

    async def get_obs_request(self, get_obs_request):
        """
        Node function. If not leader, re-directs to leader.
        Calls preprepare function.
        :return:
        """
        if self._index != self.leader:  # if self is not leader redirect to leader
            self.log("Faulty agent {} on get_obs_req. Not leader so redirecting!".format(self._index))
            raise web.HTTPTemporaryRedirect(make_url(self.leader, 'get_obs_request'))
        else:
            ####################
            # testing faulty leader
            noisy_obs = self.true_state + np.random.rand(self.true_state.size).reshape(self.true_state.shape)
            noisy_obs = np.rint(noisy_obs).astype(np.int32) * self.epsilon * 5
            self.obs = noisy_obs  # the case where only the leader's observation is faulty, but proposed obs is not;
            # everyone should still successfully commit the proposed value
            self.value_to_send = self.obs  # the case where leader's observation and proposed observation is faulty.
            ##################
            self.log("Faulty agent {} on get_obs_req. Is leader and sending faulty value {}!".format(self._index, self.value_to_send))

            if self.value_to_send is not None and not self._sent:
                self._sent = True
                #self.log("VALUE TO SEND:", self.value_to_send, self._index)
                request = {
                    'leader': self._index,
                    'data': json.dumps(self.value_to_send.tolist())}
                await self.preprepare(request)
                return web.Response()


class F_Observation(Agent):
    """
    Sends faulty observation
    """

    def __init__(self, index, n_agents, n_obs, method):
        print("Initializing Faulty Observation")
        super().__init__(index, n_agents, n_obs, method)

        ## Get agent obs (Noisy version of true obs)
    def get_obs(self):
        """
        Client function
        :return:
        """
        noisy_obs = self.true_state + np.random.rand(self.true_state.size).reshape(self.true_state.shape)
        noisy_obs = np.rint(noisy_obs).astype(np.int32) * self.epsilon * 5
        return noisy_obs


class F_Prepare(Agent):
    """
    Doesn't send prepare messages
    """

    def __init__(self, index, n_agents, method):
        print("Initializing Faulty Prepare")
        super().__init__(index, n_agents, method)

    async def prepare(self, preprepare_msg):
        self.log("Faulty agent {} on preprepare. Not doing anything!".format(self._index))
        return web.Response()


class F_Commit(Agent):
    """
    Doesn't send commit messages
    """

    def __init__(self, index, n_agents, method):
        print("Initializing Faulty Commit")
        super().__init__(index, n_agents, method)

    async def commit(self, prepare_msg):
        self.log("Faulty agent {} on commit. Not doing anything!".format(self._index))


class F_Reply(Agent):
    """
    Doesn't save commit obs and doesn't change leader
    """

    def __init__(self, index, n_agents, method):
        print("Initializing Faulty Reply")
        super().__init__(index, n_agents, method)

    async def reply(self, commit_msg):
        self.log("Faulty agent {} on reply. Not incrementing leader!".format(self._index))
        if self._closed:
            return web.Response()
        commit_msg = await commit_msg.json()
        assert (commit_msg['type'] == 'commit')
        self.log(f"Got Commit From {commit_msg['index']}, current leader is {self.leader}")
        for slot_no, data in commit_msg['proposal'].items():
            if slot_no not in self.commit_slots.keys():
                self.commit_slots[slot_no] = []
            self.commit_slots[slot_no].append(commit_msg['index'])

            if (len(self.commit_slots[slot_no]) >= 2 * self._f + 1) and (slot_no not in self.commit_sent.keys()):
                self.log("Agent {} committed".format(self._index) + str(data))
                self.commit_sent[slot_no] = True
                self.permanent_record[slot_no] = data
                self.commited_vals.append(data["data"])
                #                 np.save(f"logs/results_{self._index}.npy",  np.array(self.commit_true_counts))

                ## leader increment
                try:
                    num_sent_commits = 0
                    for slot_no, data in commit_msg['proposal'].items():  # count committed slots
                        if self.commit_sent[slot_no]:
                            num_sent_commits += 1
                    if num_sent_commits == len(
                            commit_msg['proposal'].keys()):  # if every slot has been committed, change leader
                        self.log("Due To Commit, Agent {} leader changed to {}!".format(self._index, self.leader))
                        if not self._closed:
                            self._closed = True
                            await self._session.close()
                            self.log(f"CLOSED {self._index}")
                        try:
                            self._got_response.set()
                        except:
                            pass
                except:
                    pass


class F_LeaderChange(Agent):
    """
    Doesn't participate in leader change
    """

    def __init__(self, index, n_agents, method):
        print("Initializing Faulty Leader Change")
        super().__init__(index, n_agents, method)

    async def leader_change(self, leader_change_msg):
        self.log("Faulty agent {} on leader change. Not doing anything!".format(self._index))


# robustness
#faulty_agents_list = np.array([F_Leader1, F_Prepare, F_Commit, F_LeaderChange, F_Reply])
#faulty_agents_list = np.array([F_Prepare, F_LeaderChange, F_Commit])

# for testing our method
faulty_agents_list = np.array([F_Leader2, F_Observation])
#faulty_agents_list = np.array([F_Leader2])


