from tpg_v3.program import Program
from tpg_v3.agent import Agent
import numpy as np
from tpg_v3.utils import flip
import random

"""
A team has multiple learners, each learner has a program which is executed to
produce the bid value for this learner's action.
"""
class Learner:

	idCount = 0 # unique learner id
	NumberOfModes = 2

	"""
	Create a new learner, either copied from the original or from a program or
	action. Either requires a learner, or a program/action pair.
	"""
	def __init__(self, learner=None, program=None, action=None, numRegisters=8):
		self.numRegisters = numRegisters
		self.mode = 0
		if learner is not None:
			self.program = Program(instructions=learner.program.instructions)
			self.action = learner.action
			self.numRegisters = learner.numRegisters
			self.shareIndex = learner.shareIndex
			self.mode = learner.mode
		elif program is not None and action is not None:
			self.program = program
			self.action = action
			self.shareIndex = random.randint(0, Agent.SharedRegisterGroups-1)
			self.mode = random.randint(0,Learner.NumberOfModes-1)

		if not self.isActionAtomic():
			self.action.numLearnersReferencing += 1

		self.states = []

		self.numTeamsReferencing = 0 # amount of teams with references to this

		self.id = Learner.idCount
		Learner.idCount += 1

	"""
	Get the bid value, highest gets its action selected.
	"""
	def bid(self, state, shrRegs):
		if self.mode == 0:
			return Program.execute(state, self.numRegisters,
							self.program.instructions[:,0], self.program.instructions[:,1],
							self.program.instructions[:,2], self.program.instructions[:,3],
							self.program.instructions[:,4], self.program.instructions[:,5],
							shrRegs, self.shareIndex)
		elif self.mode == 1:
			return Program.execute_vector(state, Program.sourceDims, self.numRegisters,
							self.program.instructions[:,0], self.program.instructions[:,1],
							self.program.instructions[:,2], self.program.instructions[:,3],
							self.program.instructions[:,4], self.program.instructions[:,5],
							shrRegs, self.shareIndex, Program.xShift, Program.yMask)

	"""
	Returns the action of this learner, either atomic, or requests the action
	from the action team.
	"""
	def getAction(self, state, shrRegs, visited):
		if self.isActionAtomic():
			return self.action
		else:
			return self.action.act(state, shrRegs, visited)


	"""
	Returns true if the action is atomic, otherwise the action is a team.
	"""
	def isActionAtomic(self):
		return isinstance(self.action, (int, list))

	"""
	Mutates either the program or the action or both.
	"""
	def mutate(self, pMutProg, pMutAct, pActAtom, atomics, parentTeam, allTeams,
				pDelInst, pAddInst, pSwpInst, pMutInst,
				multiActs, pSwapMultiAct, pChangeMultiAct,
				uniqueProgThresh, shrRegs, inputs=None, outputs=None):

		changed = False
		while not changed:
			# mutate the program
			if flip(pMutProg):
				changed = True
				self.program.mutate(pMutProg, pDelInst, pAddInst, pSwpInst, pMutInst,
					self.numRegisters, uniqueProgThresh, shrRegs, self.shareIndex,
					inputs=inputs, outputs=outputs)
			if flip(pMutProg):
				changed = True
				self.mode = (self.mode + 1) % Learner.NumberOfModes

			# mutate the action
			if flip(pMutAct):
				changed = True
				self.mutateAction(pActAtom, atomics, allTeams, parentTeam,
								  multiActs, pSwapMultiAct, pChangeMultiAct)
			
			if flip(pMutAct):
				changed = True
				newIdx = random.randint(0, Agent.SharedRegisterGroups-1)
				if newIdx == self.shareIndex:
					self.shareIndex = (self.shareIndex + 1) % Agent.SharedRegisterGroups

	"""
	Changes the action, into an atomic or team.
	"""
	def mutateAction(self, pActAtom, atomics, allTeams, parentTeam,
					 multiActs, pSwapMultiAct, pChangeMultiAct):
		if not self.isActionAtomic(): # dereference old team action
			self.action.numLearnersReferencing -= 1

		if flip(pActAtom): # atomic action
			if multiActs is None:
				self.action = random.choice(
								[a for a in atomics if a is not self.action])
			else:
				swap = flip(pSwapMultiAct)
				if swap or not self.isActionAtomic(): # totally swap action for another
					self.action = list(random.choice(multiActs))

				# change some value in action
				if not swap or flip(pChangeMultiAct):
					changed = False
					while not changed or flip(pChangeMultiAct):
						index = random.randint(0, len(self.action)-1)
						self.action[index] += random.gauss(0, .15)
						self.action = list(np.clip(self.action, 0, 1))
						changed = True

		else: # Team action
			self.action = random.choice([t for t in allTeams
					if t is not self.action and t is not parentTeam])

		if not self.isActionAtomic(): # add reference for new team action
			self.action.numLearnersReferencing += 1

	"""
	Saves visited states for mutation uniqueness purposes.
	"""
	def saveState(self, state, numStates=50):
		self.states.append(state)
		self.states = self.states[-numStates:]
