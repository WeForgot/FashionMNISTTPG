import pickle

import numpy as np

from tpg_v3.program import Program

"""
Simplified wrapper around a (root) team for easier interface for user.
"""
class Agent:

	SharedRegisterGroups = 8
	SharedRegisterCounts = 8

	"""
	Create an agent with a team.
	"""
	def __init__(self, team, num=1):
		self.team = team
		self.agentNum = num
		self.sharedMemory = np.zeros((Agent.SharedRegisterGroups, Agent.SharedRegisterCounts))
	
	def reset(self):
		self.sharedMemory = np.zeros((Agent.SharedRegisterGroups, Agent.SharedRegisterCounts))

	"""
	Gets an action from the root team of this agent / this agent.
	"""
	def act(self, state):
		return self.team.act(state, self.sharedMemory)

	"""
	Same as act, but with additional features. Use act for performance.
	"""
	def act2(self, state, numStates=50):
		return self.team.act2(state, self.sharedMemory, numStates=numStates)

	"""
	Give this agent/root team a reward for the given task
	"""
	def reward(self, score, task='task'):
		self.team.outcomes[task] = score

	"""
	Check if agent completed this task already, to skip.
	"""
	def taskDone(self, task):
		return task in self.team.outcomes

	"""
	Save the agent to the file, saving any relevant class values to the instance.
	"""
	def saveToFile(self, fileName):
		self.operationRange = Program.operationRange
		self.destinationRange = Program.destinationRange
		self.sourceRange = Program.sourceRange

		pickle.dump(self, open(fileName, 'wb'))

"""
Load some agent from the file, returning it and repopulate class values.
"""
def loadAgent(fileName):
	agent = pickle.load(open(fileName, 'rb'))

	Program.operationRange = agent.operationRange
	Program.destinationRange = agent.destinationRange
	Program.sourceRange = agent.sourceRange

	return agent
