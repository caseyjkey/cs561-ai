import random
import heapq
import math
import sys

class TSP_3D:
	def __init__(self, initialPopulationSize=5, numParents=2, epochs=20500):
		self.cityCount, self.cities = self.readInput()	

		self.distances = dict()
		
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
		rankHeap = []
		for i in range(size):
			path = random.sample(self.cities, self.cityCount)
			initial_population.append(path)
			heapq.heappush(rankHeap, (-1 * self.distance(path), i))

		return (initial_population, rankHeap)
	
	def rank(self, population):
		rankHeap = []
		for i, path in enumerate(population):
			heapq.heappush(rankHeap, (-1 * self.distance(path), i))
		return rankHeap

	def distance(self, path):
		path = tuple(path)
		if path in self.distances:
			return self.distances[path]

		distance = 0
		for i in range(len(path) - 1):
			x, y, z = path[i]
			x2, y2, z2 = path[i+1]	
			distance += math.sqrt((x2 - x)**2 + (y2 - y)**2 + (z2 - z)**2)
		
		x, y, z = path[-1]
		x2, y2, z2 = path[0]
		distance += math.sqrt((x2 - x)**2 + (y2 - y)**2 + (z2 - z)**2)
		distance = 1/distance

		self.distances[path] = distance

		return distance

	def createMatingPool(self, population, rankList, numParents = 2):
		matingPool = []

		heapq.heapify(rankList)	
		maxFitness = rankList[0][0] * -1 
		minFitness = rankList[-1][0] * -1
		if maxFitness != minFitness:
			# scale the ranklist to improve fitness of top individuals
			scaledRankList = []
			for rank in rankList:
				scaledRank = (-1 * math.e**((-1*rank[0] - minFitness) / (maxFitness - minFitness)), rank[1]) 
				heapq.heappush(scaledRankList, scaledRank)
			rankList = scaledRankList
		S = sum(-1 * rank[0] for rank in rankList)
		wheelPercents = [((-1 * rank[0])/S, rank[1]) for rank in rankList]
		for i in range(numParents):
			s = 0
			P = random.random()
			for percent in wheelPercents:
				s += percent[0]
				if s > P:
					matingPool.append(population[percent[1]])
					break
		return matingPool

	def evolve(self, epochs):
		population = self.initialPopulation
		rankList = self.rankList
		currentBestPath = population[rankList[0][1]]
		currentFitnessMax = rankList[0][0] * -1

		mutateFactor = 0
		for i in range(epochs):
			newPopulation = []
			newRankList = []
			# keep fittest individual
			keepTop = 1
			for i in range(keepTop):
				fittestRank = rankList[i]
				fittestIndividual = population[fittestRank[1]]
				newPopulation.append(fittestIndividual)
				heapq.heappush(newRankList, fittestRank)

			for i in range(len(population)-keepTop):
				parents = self.createMatingPool(population, rankList[keepTop:])
				start = random.randrange(len(parents[0]))
				end = random.randint(start + 1, len(parents[0]))
				child = self.crossover(parents, start, end)
				mutate = random.random()
				mutateProbability = 1/(1+mutateFactor)
				if mutate < 0.5: #mutateProbability:
					first = random.randrange(len(child))
					second = random.randrange(len(child)) 
					child[first], child[second] = child[second], child[first]
				mutateFactor += 2
				heapq.heappush(newRankList, (-1 * self.distance(child), i))
				newPopulation.append(child)
			population = newPopulation
			rankList = newRankList
			newFitnessMax = rankList[0][0] * -1
			if newFitnessMax > currentFitnessMax:
				currentBestPath = population[rankList[0][1]]
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
	file = True
	solution = TSP_3D()
	path = solution.output(file)

from datetime import datetime, timedelta
iters = 20
avgTime = timedelta(0)
avgDistance = 0
minDist = 500000000000
for i in range(iters):
	startTime = datetime.now()
	solution = TSP_3D()
	path = solution.output(file)
	dist = solution.distance(path)
	if dist < minDist:
		minDist = dist
	avgDistance += dist
	timeTaken = datetime.now() - startTime
	print(timeTaken)
	avgTime += timeTaken
avgTime = avgTime/iters
print('avg:', avgTime)
avgDistance = avgDistance/iters
print('avg dist:', avgDistance)
print('min path dist', minDist)

