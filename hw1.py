import random
import math
import sys

class TSP_3D:
	def __init__(self, initialPopulationSize=5, numParents=2, epochs=10):
		self.cityCount, self.cities = self.readInput()	
		
		self.initialPopulation, self.rankList = self.createInitialPopulation(initialPopulationSize)
		self.parents = self.createMatingPool(self.initialPopulation, self.rankList) 
		self.bestPath = self.evolve(epochs)

	def readInput(self) -> tuple[int, list[tuple[int, int, int]]]:
		cities = []
		with open('input.txt', 'r') as file:
			cityCount = int(file.readline().strip())
			for city in file.readlines():
				city = tuple([int(coord) for coord in city.strip().split()])
				cities.append(city)
			
		return (cityCount, cities)

	def createInitialPopulation(self, size: int) -> tuple[list[tuple[int, float]], list[tuple[int, int, int]]]:
		initial_population = []
		for i in range(size):
			path = random.sample(self.cities, self.cityCount)
			initial_population.append(path)
				
		rankList = self.rank(initial_population)
		return (initial_population, rankList)
	
	def rank(self, population: [list[list[tuple[int, int, int]]]]) -> list[tuple[int, float]]:
		rankList = []
		for i, path in enumerate(population):
			rankList.append((i, self.distance(path)))
		rankList.sort(key=lambda rank: rank[1], reverse=True)
		return rankList		

	def distance(self, path: list[tuple[int, int, int]]) -> float:
		distance = 0
		for i in range(len(path) - 1):
			x, y, z = path[i]
			x2, y2, z2 = path[i+1]	
			distance += math.sqrt((x2 - x)**2 + (y2 - y)**2 + (z2 - z)**2)
		x, y, z = path[-1]
		x2, y2, z2 = path[0]
		distance += math.sqrt((x2 - x)**2 + (y2 - y)**2 + (z2 - z)**2)

		return 1/distance

	def createMatingPool(self, population: list[tuple[int, int, int]], rankList: list[tuple[int, int]], numParents = 2) -> list:
		matingPool = []

		for i in range(numParents):
			S = sum(rank[1] for rank in rankList)
			s = 0
			P = random.random()
			for rank in rankList:
				s += rank[1]/S
				if s > P:	
					matingPool.append(self.initialPopulation[rank[0]])
					P = random.random()
					s = 0
					break

		return matingPool

	def evolve(self, epochs=10):
		population = self.initialPopulation
		rankList = self.rankList
		bestRank = self.rankList[0]
		for i in range(epochs):
			while True:
				newPopulation = []
				for i in range(len(population)):
					parents = self.createMatingPool(population, rankList)
					start = random.randint(0, len(parents[0])-1)
					end = random.randint(start + 1, len(parents[0]))
					child = self.crossover(parents, start, end)
					newPopulation.append(child)
				population = newPopulation
				rankList = self.rank(population)
				if rankList[0][1] >= bestRank[1] and rankList[0][0] != bestRank[0]:
					bestRank = rankList[0]
					print('new best', population[bestRank[0]])
					print('fitness:', bestRank[1])
					break
		
		bestPath = population[rankList[0][0]]

		return bestPath
	
	def crossover(self, parents, start, end):
		p1, p2 = parents[0], parents[1]
		child = p2[0:start] + p1[start:end+1] + p2[end+1:]

		if (missing := set(p1).difference(set(child))):
			child = [city if city not in child[:i] else missing.pop() for i, city in enumerate(child)]
		return child
	
	def output(self, file=False):
		path = self.bestPath
		path.append(path[0])

		output = [' '.join([str(coord) for coord in city]) for city in path]

		if file:
			with open('output.txt', 'w') as file:
				file.writelines(output)
		
		else:
			for city in output:
				print(city)
			
		return
			
epochs = 10000
if len(sys.argv) > 1:
	epochs = int(sys.argv[1])
solution = TSP_3D(epochs=epochs)
solution.output()
print(solution.distance([(120,199,34), (199,173,30), (175,53,76),(144,39,130),(173,101,186),(153,196,97),(137,199,93)]))
