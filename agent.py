import numpy as np

class Agent:

    def __init__(self,):
        pass

    def get_obs(self):
        """
        Client function
        :return:
        """
        pass

    def send_obs_request(self):
        """
        Client function (sends requests)
        :return:
        """
        pass

    def get_obs_request(self):
        """
        Node function. If not leader, re-directs to leader.
        Calls preprepare function.
        :return:
        """
        pass

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


