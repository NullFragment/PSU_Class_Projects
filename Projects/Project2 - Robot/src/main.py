import math
import random

_author_ = 'NullFragment'


class Board:
	def __init__(self, rows, columns, gold):
		self.rows = rows
		self.columns = columns
		self.cells = [[Cell(i, j) for i in range(self.columns)] for j in range(self.rows)]
		self.goldCount = 0
		self.maxGold = gold
	
	def AddGold(self):
		while self.goldCount < self.maxGold:
			i = random.randint(0, self.rows - 1)
			j = random.randint(0, self.columns - 1)
			k = random.randint(0, 1)
			if self.cells[i][j].contents == [0, 0, 0] and k == 1:
				self.cells[i][j].AddGold()
				self.goldCount += 1
	
	def PrintBoard(self):
		for i in range(self.rows):
			print("| ", end='')
			for j in range(self.columns):
				if self.cells[i][j].contents[0] == 1:
					print("G", end='')
				if self.cells[i][j].contents[1] == 1:
					print("R", end='')
				if self.cells[i][j].contents[2] == 1:
					print("B", end='')
				if self.cells[i][j].contents == [0, 0, 0]:
					print("E", end='')
				if j < self.columns - 1:
					print(" | ", end='')
				else:
					print(" ", end='')
			print("|")


class Cell:
	def __init__(self, row, column):
		self.row = row
		self.column = column
		self.location = [row, column]
		self.contents = [0, 0, 0]
	
	# Contents: [ Gold, Robot, Bomb ]
	# 0 = not occupying, 1 = occupying
	
	def AddGold(self):
		self.contents[0] = 1
	
	def AddRobot(self):
		self.contents[1] = 1
	
	def AddBomb(self):
		self.contents[2] = 1
	
	def RemoveGold(self):
		self.contents[0] = 0
	
	def RemoveRobot(self):
		self.contents[1] = 0
	
	def RemoveBomb(self):
		self.contents[2] = 0


class Robot:
	def __init__(self, board, row, column):
		self.board = board
		self.cell = self.board.cells[row][column]
		self.currentGold = 0
		self.cell.AddRobot()
	
	def GrabGold(self):
		if self.cell.contents[0] == 1:
			self.currentGold += 1
			self.cell.RemoveGold()
			self.board.goldCount -= 1
			print("Gold Reserves at: ", self.currentGold)
		# else: print("No gold found")
	
	def Move(self, rowMove, columnMove):
		if self.cell.location[0] + columnMove < self.board.columns and self.cell.location[
			1] + rowMove < self.board.rows and self.cell.location[0] + columnMove > -1 and self.cell.location[
			1] + rowMove > -1:
			self.cell.RemoveRobot()
			self.cell = self.board.cells[self.cell.location[1] + rowMove][self.cell.location[0] + columnMove]
			self.cell.AddRobot()
		else:
			print("ERROR: Board Boundaries Reached")


class Bomb:
	def __init__(self, board, row, column):
		self.board = board
		self.cell = self.board.cells[row][column]
		self.currentGold = 0
		self.cell.AddBomb()
	
	def Move(self, rowMove, columnMove):
		if self.cell.location[0] + columnMove < self.board.columns and self.cell.location[
			1] + rowMove < self.board.rows and self.cell.location[0] + columnMove > -1 and self.cell.location[
			1] + rowMove > -1:
			self.cell.RemoveBomb()
			self.cell = self.board.cells[self.cell.location[1] + rowMove][self.cell.location[0] + columnMove]
			self.cell.AddBomb()
		else:
			print("ERROR: Board Boundaries Reached")


class Simulation:
	def __init__(self, rows, columns, gold):
		self.simBoard = Board(rows, columns, gold)
		self.rows = rows - 1
		self.columns = columns - 1
		robotStart = [random.randint(0, 3), random.randint(0, 3)]
		bombStart = [random.randint(0, 3), random.randint(0, 3)]
		while robotStart[0] == bombStart[0] and robotStart[1] == bombStart[1]:
			print("looping")
			bombStart = [random.randint(0, 3), random.randint(0, 3)]
		self.simRobot = Robot(self.simBoard, robotStart[0], robotStart[1])
		self.simBomb = Bomb(self.simBoard, bombStart[0], bombStart[1])
		self.simBoard.AddGold()
		self.simBoard.PrintBoard()
	
	def GetNewDirection(self, currentAxisLocation):
		if currentAxisLocation > 0 and currentAxisLocation < 3:
			return random.randint(-1, 1)
		elif currentAxisLocation == 3:
			return random.randint(-1, 0)
		elif currentAxisLocation == 0:
			return random.randint(0, 1)
	
	def MoveCharacterRandom(self, character):
		startMove = character.cell.location
		moveX = self.GetNewDirection(character.cell.location[1])
		moveY = self.GetNewDirection(character.cell.location[0])
		while moveX == 0 or moveY == 0:
			moveX = self.GetNewDirection(character.cell.location[1])
			moveY = self.GetNewDirection(character.cell.location[0])
		character.Move(moveX, moveY)
		endMove = character.cell.location
		print(type(character).__name__, ": ", startMove, " -> ", endMove)
	
	def MoveCharacter(self, character, moveX, moveY):
		startMove = character.cell.location
		character.Move(moveX, moveY)
		endMove = character.cell.location
		print(type(character).__name__, ": ", startMove, " -> ", endMove)
	
	def CheckForBomb(self):
		if self.simRobot.cell.location == self.simBomb.cell.location:
			print("The robot was blown up!")
			exit(4)
	
	def MoveRobotToStart(self):
		while self.simRobot.cell.location != [0, 0]:
			if self.simRobot.cell.location[1] > 0:
				robotMoveX = -1
			else:
				robotMoveX = 0
			if self.simRobot.cell.location[0] > 0:
				robotMoveY = -1
			else:
				robotMoveY = 0
			self.MoveCharacter(self.simRobot, robotMoveX, robotMoveY)
			self.MoveCharacterRandom(self.simBomb)
			self.simBoard.PrintBoard()
			self.CheckForBomb()
	
	def RunSimulation(self):
		self.MoveRobotToStart()
		moveLeft = False
		while self.simBoard.goldCount > 0:
			self.simRobot.GrabGold()
			if self.simRobot.cell.location[0] < self.columns and moveLeft == False:
				robotMoveX = 1
				robotMoveY = 0
			elif self.simRobot.cell.location[0] > 0 and moveLeft == True:
				robotMoveX = -1
				robotMoveY = 0
			else:
				robotMoveY = 1
				robotMoveX = 0
				moveLeft = not moveLeft
			self.MoveCharacter(self.simRobot, robotMoveY, robotMoveX)
			self.MoveCharacterRandom(self.simBomb)
			self.simBoard.PrintBoard()
			self.CheckForBomb()
		print("The robot collected all of the gold!")
		exit(3)


robotSimulation = Simulation(4, 4, 4)
robotSimulation.RunSimulation()
