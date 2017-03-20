import math
import random

_author_ = 'NullFragment'


class Board:
	# This class creates the game board as a 2D array of Cell classes, distributes the goald to the board, and
	# prints the board layout with each cell's contents.
	# The board is initalized by giving it the number of rows and columns to make as well as the maximum amount of gold
	# allowed on the board when using the AddGold() function.
	
	def __init__(self, rows, columns, gold):
		self.rows = rows
		self.columns = columns
		self.cells = [[Cell(i, j) for i in range(self.columns)] for j in range(self.rows)]
			# self.cells loops through a 2D array of [i = rows][j = columns]
		self.goldCount = 0
		self.maxGold = gold
	
	def AddGold(self):
		# Adds a gold piece to the board by pseudo-randomly picking a cell and deciding whether or not the cell should
		# get a piece of gold (variable k). If the cell is empty and k is true, it adds a piece of gold to that cell.
		while self.goldCount < self.maxGold:
			i = random.randint(0, self.rows - 1)
			j = random.randint(0, self.columns - 1)
			k = random.randint(0, 1)
			if self.cells[i][j].contents == [0, 0, 0] and k == 1:
				self.cells[i][j].AddGold()
				self.goldCount += 1
	
	def PrintBoard(self):
		# This function prints the game board cell by cell.
		# G = Cell has gold
		# R = Robot is in the cell
		# B = Bomb is in the cell
		# E = Cell is empty
		# Cells are formatted as: "| X |"
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
	# The Cell class contains the coordinates of each cell (location, row, column) and the cell's contents.
	# Contents: [ Gold, Robot, Bomb ]
	# 0 = not occupying, 1 = occupying
	# It also contains methods to add or remove Gold, the Robot or the Bomb from each cell.
	# It is initialized by giving it a row and column for it's location and sets inital contents to empty.
	def __init__(self, row, column):
		self.row = row
		self.column = column
		self.location = [row, column]
		self.contents = [0, 0, 0]
	
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
	# The robot class is initalized by assigning it to a game board (class Board) and giving it an initial position
	# on that board. It also sets an initial gold amount of 0.
	# During initialization, it assigns an internal Cell variable that points to the initial position on the game board
	# and adds itself to that cell using th AddRobot() function.
	
	def __init__(self, board, row, column):
		self.board = board
		self.cell = self.board.cells[row][column]
		self.currentGold = 0
		self.cell.AddRobot()
	
	def GrabGold(self):
		# This function checks to see if the current cell contains gold. If it does, it increments it's gold counter by
		# 1, removes the gold from the cell, and decrements the board's total gold by 1.
		if self.cell.contents[0] == 1:
			self.currentGold += 1
			self.cell.RemoveGold()
			self.board.goldCount -= 1
			print("Gold Reserves at: ", self.currentGold)
	
	def Move(self, rowMove, columnMove):
		# Moves the robot by X,Y vector as long as the vector is within the board's boundaries.
		# If the movement is allowed, the robot removes itself from the current cell using RemoveRobot(), updates
		# it's cell to the new location and uses AddRobot() to update the game board.
		
		if self.cell.location[0] + columnMove < self.board.columns and self.cell.location[
			1] + rowMove < self.board.rows and self.cell.location[0] + columnMove > -1 and self.cell.location[
			1] + rowMove > -1:
			self.cell.RemoveRobot()
			self.cell = self.board.cells[self.cell.location[1] + rowMove][self.cell.location[0] + columnMove]
			self.cell.AddRobot()
		else:
			print("ERROR: Board Boundaries Reached")


class Bomb:
	# The bomb class is initalized by assigning it to a game board (class Board) and giving it an initial position
	# on that board.
	# During initialization, it assigns an internal Cell variable that points to the initial position on the game board
	# and adds itself to that cell using th AddBomb() function.
	
	def __init__(self, board, row, column):
		self.board = board
		self.cell = self.board.cells[row][column]
		self.cell.AddBomb()
	
	def Move(self, rowMove, columnMove):
		# Moves the bomb by X,Y vector as long as the vector is within the board's boundaries.
		# If the movement is allowed, the robot removes itself from the current cell using RemoveBomb(), updates
		# it's cell to the new location and uses AddBomb() to update the game board.
		
		if self.cell.location[0] + columnMove < self.board.columns and self.cell.location[
			1] + rowMove < self.board.rows and self.cell.location[0] + columnMove > -1 and self.cell.location[
			1] + rowMove > -1:
			self.cell.RemoveBomb()
			self.cell = self.board.cells[self.cell.location[1] + rowMove][self.cell.location[0] + columnMove]
			self.cell.AddBomb()
		else:
			print("ERROR: Board Boundaries Reached")


class Simulation:
	# The simulation class handles actually running the simulation of the robot and bomb moving around the board.
	# It is initalized by giving a number of rows and columns that the board will consist of as well as the maximum
	# amount of gold on the board.
	# The initializer creates the board (simBoard) and determines a random starting location for the robot and then tbe
	# bomb. If the bomb location is the same as the robot, the bomb location is randomized until they are not the same.
	# The gold is then populated and the simulation begins.
	#
	# This simulation makes the robot head to the starting position [0,0] and then comb through each column before
	# moving down a row and then it combs the other way. At maximum the robot will move sqrt(Rows^2 + Columns^2) +
	# Rows * Columns times if it starts in the furthest point from the origin [0,0]
	#
	# The robot will go [0,0] -> [0, columns] and then [1, columns] -> [1, 0] in it's comb to find the gold.
	
	def __init__(self, rows, columns, gold):
		self.simBoard = Board(rows, columns, gold)
		self.rows = rows - 1
		self.columns = columns - 1
		robotStart = [random.randint(0, 3), random.randint(0, 3)]
		bombStart = [random.randint(0, 3), random.randint(0, 3)]
		while robotStart[0] == bombStart[0] and robotStart[1] == bombStart[1]:
			# Ensures bomb and robot do not start in the same cell.
			bombStart = [random.randint(0, 3), random.randint(0, 3)]
		self.simRobot = Robot(self.simBoard, robotStart[0], robotStart[1])
		self.simBomb = Bomb(self.simBoard, bombStart[0], bombStart[1])
		self.simBoard.AddGold()
		self.simBoard.PrintBoard()
	
	def GetNewDirection(self, currentAxisLocation):
		# Gets a random direction for moving.
		# !!! CURRENTLY ONLY HANDLES 4x4 GRIDS! !!!
		if currentAxisLocation > 0 and currentAxisLocation < 3:
			return random.randint(-1, 1)
		elif currentAxisLocation == 3:
			# If character is at the far edge of the grid, it can only move nowhere or inward
			return random.randint(-1, 0)
		elif currentAxisLocation == 0:
			# If character is at the close edge of the grid, it can only move nowhere or outward
			return random.randint(0, 1)
	
	def MoveCharacterRandom(self, character):
		# Randomly moves a given character and prints what character is moving, from which cell it is moving and to
		# which cell it is moving.
		
		startMove = character.cell.location
		moveX = self.GetNewDirection(character.cell.location[1])
		moveY = self.GetNewDirection(character.cell.location[0])
		while moveX == 0 or moveY == 0:
			# Ensures character does not stay still
			moveX = self.GetNewDirection(character.cell.location[1])
			moveY = self.GetNewDirection(character.cell.location[0])
		character.Move(moveX, moveY)
		endMove = character.cell.location
		print(type(character).__name__, ": ", startMove, " -> ", endMove)
	
	def MoveCharacter(self, character, moveX, moveY):
		# Moves a character in a specified direction and prints what character is moving, from which cell it is
		# moving and to which cell it is moving.
		
		startMove = character.cell.location
		character.Move(moveX, moveY)
		endMove = character.cell.location
		print(type(character).__name__, ": ", startMove, " -> ", endMove)
	
	def CheckForBomb(self):
		# Checks if the the cell the robot is in also contains the bomb.
		# If this is true, the game ends.
		if self.simRobot.cell.location == self.simBomb.cell.location:
			print("The robot was blown up!")
			exit(4)
	
	def MoveRobotToStart(self):
		# Runs the simulation with the goal of moving the robot to [0,0]
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
			# While there is still gold on the board, the robot will comb through each row, cell by cell, first moving
			# right, until it reaches the end of the row. It then moves down to the next row and moves to the left
			# cell by cell toward the beginning of the row.
			self.simRobot.GrabGold()
			# Checks to get gold every cell.
			if self.simRobot.cell.location[0] < self.columns and moveLeft == False:
				# Queues the robot to move the right cell by cell until it reaches the end of the row.
				robotMoveX = 1
				robotMoveY = 0
			elif self.simRobot.cell.location[0] > 0 and moveLeft == True:
				# Queues the robot to move to the left cell by cell until it reaches the start of the row.
				robotMoveX = -1
				robotMoveY = 0
			else:
				# Queues the robot to move down one row and changes the direction of the robot's movement
				robotMoveY = 1
				robotMoveX = 0
				moveLeft = not moveLeft
			self.MoveCharacter(self.simRobot, robotMoveY, robotMoveX)
				# Moves the robot
			self.MoveCharacterRandom(self.simBomb)
				# Moves the bomb
			self.simBoard.PrintBoard()
				# Prints the game board
			self.CheckForBomb()
				# Checks if the bomb moved onto the robot
		# If the robot successfully collects all of the gold before the robot blows it up, the game ends successfully.
		print("The robot collected all of the gold!")
		exit(3)


robotSimulation = Simulation(4, 4, 4)
robotSimulation.RunSimulation()
