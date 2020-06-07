from agent import *


class F_Leader1(Agent):
    """
    Doesn't send observation request
    """

    def __init__(self, index, n_agents):
        print("Initializing Faulty Leader 1")
        super().__init__(index, n_agents)

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

    def __init__(self, index, n_agents):
        print("Initializing Faulty Leader 2")
        super().__init__(index, n_agents)

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
            self.obs = 1  # the case where only the leader's observation is faulty, but proposed obs is not;
            # everyone should still successfully commit the proposed value
            self.value_to_send = self.obs  # the case where leader's observation and proposed observation is faulty.
            ##################
            self.log("Faulty agent {} on get_obs_req. Is leader and sending faulty value {}!".format(self._index, self.value_to_send))

            if self.value_to_send is not None and not self._sent:
                self._sent = True
                #self.log("VALUE TO SEND:", self.value_to_send, self._index)
                request = {
                    'leader': self._index,
                    'data': self.value_to_send}
                await self.preprepare(request)
                return web.Response()


class F_Prepare(Agent):
    """
    Doesn't send prepare messages
    """

    def __init__(self, index, n_agents):
        print("Initializing Faulty Prepare")
        super().__init__(index, n_agents)

    async def prepare(self, preprepare_msg):
        self.log("Faulty agent {} on preprepare. Not doing anything!".format(self._index))
        return web.Response()


class F_Commit(Agent):
    """
    Doesn't send commit messages
    """

    def __init__(self, index, n_agents):
        print("Initializing Faulty Commit")
        super().__init__(index, n_agents)

    async def commit(self, prepare_msg):
        self.log("Faulty agent {} on commit. Not doing anything!".format(self._index))


class F_Reply(Agent):
    """
    Doesn't save commit obs and doesn't change leader
    """

    def __init__(self, index, n_agents):
        print("Initializing Faulty Reply")
        super().__init__(index, n_agents)

    async def reply(self, commit_msg):
        self.log("Faulty agent {} on reply. Not doing anything!".format(self._index))


class F_LeaderChange(Agent):
    """
    Doesn't participate in leader change
    """

    def __init__(self, index, n_agents):
        print("Initializing Faulty Leader Change")
        super().__init__(index, n_agents)

    async def leader_change(self, leader_change_msg):
        self.log("Faulty agent {} on leader change. Not doing anything!".format(self._index))


faulty_agents_list = np.array([F_Leader1, F_Leader2, F_Prepare, F_Commit, F_LeaderChange])
# faulty_agents_list = np.array([F_Leader1,  F_Commit])
#faulty_agents_list = np.array([F_Prepare,  F_Commit])


# Single timestep:
# -FPrepare, FCommit, FLeader1, FLeader2

# Multi timestep
# -FLeaderChange, FReply