import random
import math
import sys

class TSP_3D:
	def __init__(self, initialPopulationSize=5, numParents=2, epochs=100000):
		self.cityCount, self.cities = self.readInput()	
		
		self.initialPopulation, self.rankList = self.createInitialPopulation(initialPopulationSize)
		self.bestPath = self.evolve(epochs)

	def readInput(self):
		cities = []
		with open('input.txt', 'r') as file:
			cityCount = int(file.readline().strip())
			for city in file.readlines():
				city = tuple([int(coord) for coord in city.strip().split()])
				cities.append(city)
			
		return (cityCount, cities)

	def createInitialPopulation(self, size):
		initial_population = []
		for i in range(size):
			path = random.sample(self.cities, self.cityCount)
			initial_population.append(path)
				
		rankList = self.rank(initial_population)
		return (initial_population, rankList)
	
	def rank(self, population):
		rankList = []
		for i, path in enumerate(population):
			rankList.append((i, self.distance(path)))
		rankList.sort(key=lambda rank: rank[1], reverse=True)
		return rankList		

	def distance(self, path):
		distance = 0
		for i in range(len(path) - 1):
			x, y, z = path[i]
			x2, y2, z2 = path[i+1]	
			distance += math.sqrt((x2 - x)**2 + (y2 - y)**2 + (z2 - z)**2)
		
		x, y, z = path[-1]
		x2, y2, z2 = path[0]
		distance += math.sqrt((x2 - x)**2 + (y2 - y)**2 + (z2 - z)**2)

		return 1/distance

	def createMatingPool(self, population, rankList, numParents = 2):
		matingPool = []

		for i in range(numParents):
			S = sum(rank[1] for rank in rankList)
			s = 0
			P = random.random()
			for rank in rankList:
				s += rank[1]/S
				if s > P:
					matingPool.append(population[rank[0]])
					P = random.random()
					s = 0
					break

		return matingPool

	def evolve(self, epochs=1):
		population = self.initialPopulation
		rankList = self.rankList
		currentBestPath = population[rankList[0][0]]
		currentFitnessMax = rankList[0][1]

		for i in range(epochs):
			newPopulation = []
			for i in range(len(population)):
				parents = self.createMatingPool(population, rankList)
				start = random.randint(0, len(parents[0])-1)
				end = random.randint(start + 1, len(parents[0]))
				child = self.crossover(parents, start, end)
				if random.random() > 0.5:
					first = random.randrange(len(child))
					second = random.randrange(len(child))
					child[first], child[second] = child[second], child[first]
				newPopulation.append(child)
			population = newPopulation
			rankList = self.rank(population)
			newBestPath = population[rankList[0][0]]
			newFitnessMax = rankList[0][1]
			if newFitnessMax >= currentFitnessMax and currentBestPath != newBestPath:
				currentBestPath = newBestPath
				currentFitnessMax = newFitnessMax

		return currentBestPath
	
	def crossover(self, parents, start, end):
		p1, p2 = parents[0], parents[1]
		child = p2[0:start] + p1[start:end+1] + p2[end+1:]

		missing = set(p1).difference(set(child))
		if missing:
			child = [city if city not in child[:i] else missing.pop() for i, city in enumerate(child)]
		return child
	
	def output(self, file=True):
		path = self.bestPath
		path.append(path[0])

		output = [' '.join([str(coord) for coord in city]) for city in path]

		if file:
			with open('output.txt', 'w') as file:
				file.writelines([line + '\n' for line in output])
		
		else:
			for city in output:
				print(city)
			
		return path
			
if __name__ == '__main__':
	# epochs = 100000
	# if len(sys.argv) > 1:
	#	epochs = int(sys.argv[1])
	solution = TSP_3D()
	path = solution.output()
