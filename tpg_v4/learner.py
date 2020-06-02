from tpg_v4.program import Program
from tpg_v4.agent import Agent
import numpy as np
from tpg_v4.utils import flip
import random

"""
A team has multiple learners, each learner has a program which is executed to
produce the bid value for this learner's action.
"""
class Learner:

	idCount = 0 # unique learner id
	SourceDimensions = [28,28]
	SourceKernelSize = 3

	"""
	Create a new learner, either copied from the original or from a program or
	action. Either requires a learner, or a program/action pair.
	"""
	def __init__(self, learner=None, program=None, action=None, numRegisters=8):
		if learner is not None:
			self.program = Program(instructions=learner.program.instructions)
			self.action = learner.action
			self.registers = np.zeros(len(learner.registers), dtype=float)
			self.shareIndex = learner.shareIndex
			self.obsSrc = learner.obsSrc
		elif program is not None and action is not None:
			self.program = program
			self.action = action
			self.registers = np.zeros(numRegisters, dtype=float)
			self.shareIndex = random.randint(0, Agent.SharedRegisterGroups-1)
			self.obsSrc = np.zeros(len(Learner.SourceDimensions), dtype=np.int32)
			for idx in range(len(Learner.SourceDimensions)):
				self.obsSrc[idx] = random.randint(0,Learner.SourceDimensions[idx] - Learner.SourceKernelSize - 1)

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
		'''
		For right now I am going to say the sub-observation indexing starts in the top left of the kernel and goes for the kernel size (X is the coordinate in self.obsSrc)
		X00
		000
		000
		I need to change it to be
		000
		0X0
		000
		but I am having a brainfart right now so just gonna leave it be to get SOMETHING IN. The reason we should do it the latter way is because the former deals with the top left
		differently than the bottom right. The fix is above by subtracting half the kernel size from the randomly sampled source coordinates but that is kinda jank
		'''
		sliceObj = tuple(slice(idl,idl+Learner.SourceKernelSize) for idl in self.obsSrc)
		newObs = state[sliceObj]
		self.registers.fill(0)
		#Program.execute(state, self.registers,
		Program.execute(newObs, self.registers,
						self.program.instructions[:,0], self.program.instructions[:,1],
						self.program.instructions[:,2], self.program.instructions[:,3],
						self.program.instructions[:,4], self.program.instructions[:,5],
						shrRegs, self.shareIndex)

		return self.registers[0]

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
					len(self.registers), uniqueProgThresh, shrRegs, self.shareIndex,
					inputs=inputs, outputs=outputs)

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
				self.shareIndex = newIdx
			
			if flip(pMutAct):
				changed = True
				posShift = np.random.randint(-1, 1, len(Learner.SourceDimensions))
				while np.count_nonzero(posShift) == 0:
					posShift = np.random.randint(-1, 1, len(Learner.SourceDimensions))
				self.obsSrc = np.mod(posShift + self.obsSrc, Learner.SourceDimensions)


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
