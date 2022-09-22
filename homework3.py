import random
import heapq
import math
import sys

class TSP_3D:
	def __init__(self, solve=False, initialPopulationSize=2, numParents=2, epochs=500):
		self.distances = dict()

		if solve:
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
		if distance > 0:
			distance = 1/distance

		self.distances[path] = distance

		return distance

	def createMatingPool(self, population, rankList, numParents = 2):
		matingPool = []
		for i in range(numParents):
			fittestIndex = heapq.heappop(rankList)[1]
			matingPool.append(population[fittestIndex])
		"""
		maxFitness = rankList[0][0] * -1 
		minFitness = max(rankList)[0] * -1
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
		"""
		return matingPool

	def evolve(self, epochs):
		population = self.initialPopulation
		rankList = self.rankList
		currentBestPath = population[rankList[0][1]]
		currentFitnessMax = rankList[0][0] * -1

		for j in range(epochs):
			newPopulation = []
			newRankList = []

			parents = self.createMatingPool(population, rankList)
			parents[0] = self.locallyOptimized(parents[0])
			parents = self.populationReformulation(parents)
			children = self.crossover(parents)
			for i, child in enumerate(children):
				newChild = self.rearrangement(child)
				if newChild != child:
					children[i] = newChild
			
			for parent in parents:
				children += self.multiMutate(parent)			

			for i, child in enumerate(children):
				heapq.heappush(newRankList, (-1 * self.distance(child), i))
			population = children
			rankList = newRankList

			newFitnessMax = rankList[0][0] * -1
			if newFitnessMax > currentFitnessMax:
				currentBestPath = population[rankList[0][1]]
				currentFitnessMax = newFitnessMax
		return currentBestPath
	
	def crossover(self, parents, start=None, end=None, mid=None):

		children = []

		start = random.randrange(len(parents[0])) if not start else start
		end = random.randint(start + 1, len(parents[0])) if not end else end
		children += self.twoPointSwappedInvertedCrossover(parents, start, end)		
		mid = random.randrange(len(parents[0])) if not mid else mid
		children += self.onePointSwappedInvertedCrossover(parents, mid)


		'''
		child = p2[0:start] + p1[start:end+1] + p2[end+1:]
		missing = set(p1).difference(set(child))
		if missing:
			child = [city if city not in child[:i] else missing.pop() for i, city in enumerate(child)]
		'''
		return children
	
	def populationReformulation(self, parents):
		newParents = []

		mid = len(parents[0]) // 2
		for parent in parents:
			newParents.append(parent[mid::-1] + parent[:mid:-1])

		return newParents

	def rearrangement(self, path):
		maxDist = [0, 0]
		for i in range(len(path)-1):
			x, y, z = path[i]
			x2, y2, z2 = path[i+1]	
			distance = math.sqrt((x2 - x)**2 + (y2 - y)**2 + (z2 - z)**2)
			if distance > maxDist[0]:
				maxDist = [distance, i]

		s1 = path[:]
		s1[maxDist[1]], s1[0] = s1[0], s1[maxDist[1]]
		s2 = path[:]
		s2[maxDist[1]], s2[len(path) // 2] = s2[len(path) // 2], s2[maxDist[1]]
		s3 = path[:]
		s3[maxDist[1]], s3[len(path)-1] = s3[len(path)-1], s3[maxDist[1]]
		maxDist = [-1, None]
		for S in [path, s1, s2, s3]:
			dist = self.distance(S)
			if dist > maxDist[0]:
				maxDist = [dist, S]
		return maxDist[1]

	def twoPointSwappedInvertedCrossover(self, parents, start, end):
		p1, p2 = parents[0], parents[1]
		p1Head, p2Head = p1[0:start], p2[0:start]
		p1Tail, p2Tail = p1[end+1:], p2[end+1:]
		mid1 = [city2 for city2 in p2 if city2 not in (p1Head+p1Tail)]
		mid2 = [city1 for city1 in p1 if city1 not in (p2Head+p2Tail)]

		o1 = p1Tail[::-1] + mid1 + p1Head[::-1]
		o2 = p2Tail[::-1] + mid2 + p2Head[::-1]
		o3 = p1Head[::-1] + mid1 + p1Tail[::-1]
		o4 = p2Head[::-1] + mid2 + p2Tail[::-1]

		children = [o1, o2, o3, o4]
		return children

	def onePointSwappedInvertedCrossover(self, parents, mid):
		p1, p2 = parents[0], parents[1]
		p1Head = p1[0:mid]
		p2Tail = [city2 for city2 in p2 if city2 not in p1Head]
		o1 = p1Head[::-1] + p2Tail
		o2 = p2Tail + p1Head[::-1]

		p2Head = p2[0:mid]
		p1Tail = [city1 for city1 in p1 if city1 not in p2Head]
		o3 = p2Head[::-1] + p1Tail
		o4 = p1Tail + p2Head[::-1]

		p1Tail = p1[mid-1:]
		p2Head = [city2 for city2 in p2 if city2 not in p1Tail]
		o5 = p1Tail[::-1] + p2Head
		o6 = p2Head + p1Tail[::-1]

		p2Tail = p2[mid-1:]
		p1Head = [city1 for city1 in p1 if city1 not in p2Tail]
		o7 = p2Tail[::-1] + p1Head
		o8 = p1Head + p2Tail[::-1]

		children = [o1, o2, o3, o4, o5, o6, o7, o8]
		return children

	def multiMutate(self, parent):
		mutations = []
		c1 = random.randrange(len(parent))
		c2 = random.randrange(len(parent))
		mutated = parent
		if c1 < len(parent) - 1:
			mutated[c1], mutated[c1+1] = mutated[c1+1], mutated[c1]
			mutations.append(mutated)

		mutated = parent
		if c1 > 0:
			mutated[c1], mutated[c1-1] = mutated[c1-1], mutated[c1]
			mutations.append(mutated)

		mutated = parent
		if c1 < len(parent) - 2:
			mutated[c1], mutated[c1+2] = mutated[c1+2], mutated[c1]
			mutations.append(mutated)

		mutated = parent
		if c1 > 1:
			mutated[c1], mutated[c1-2] = mutated[c1-2], mutated[c1]
			mutations.append(mutated)

		mutated = parent
		if c2 < len(parent) - 1:
			mutated[c2], mutated[c2+1] = mutated[c2+1], mutated[c2]
			mutations.append(mutated)

		mutated = parent
		if c2 > 0:
			mutated[c2], mutated[c2-1] = mutated[c2-1], mutated[c2]
			mutations.append(mutated)

		mutated = parent
		if c2 < len(parent) - 2:
			mutated[c2], mutated[c2+2] = mutated[c2+2], mutated[c2]
			mutations.append(mutated)

		mutated = parent
		if c1 > 1:
			mutated[c2], mutated[c2-2] = mutated[c2-2], mutated[c2]
			mutations.append(mutated)


		mutated = parent
		mutated[c1], mutated[c2] = mutated[c2], mutated[c1]
		mutations.append(mutated)

		mutated = parent
		mutated[0], mutated[len(parent)-1] = mutated[len(parent)-1], mutated[0]
		mutations.append(mutated)

		return mutations

	def locallyOptimized(self, currentPath):
		if len(currentPath) <= 5:
			subtourLength = len(currentPath[1:len(currentPath)-1])
		else:
			subtourLength  = random.randrange(3, len(currentPath)//2)
		start = random.randrange(0, len(currentPath) - subtourLength)
		subtour = currentPath[start:start+subtourLength]
		if len(subtour) <= 1:
			return currentPath
		currentEvals = 0
		maxEvals = 1500
		bestFitness = self.distance(subtour)
		bestPath = subtour
		while currentEvals < maxEvals:
			moved = False
			for newPath in self.reverse(bestPath):
				if currentEvals >= maxEvals:
					break

				newFitness = self.distance(newPath)
				currentEvals += 1
				if newFitness > bestFitness:
					bestFitness = newFitness
					bestPath = newPath
					moved = True
			if not moved:
				# we are at a local maximum
				break

		bestPath = currentPath[:start] + bestPath + currentPath[start+subtourLength:]
		if len(bestPath) != len(currentPath):
			print('BORKED best, current')
			print(bestPath)
			print(currentPath)
		
		return bestPath

	def reverse(self, path):
		for i, j in self.pairs(len(path)):
			if i != j:
				newPath = path[:]
				if i < j:
					newPath[i:j+1] = reversed(path[i:j+1])
				else:
					newPath[i+1:] = reversed(path[:j])
					newPath[:j] = reversed(path[i+1:])
				if newPath != path:
					yield newPath
		
	def pairs(self, size):
		l1, l2 = list(range(size)), list(range(size))
		random.shuffle(l1)
		random.shuffle(l2)
		for i in l1:
			for j in l2:
				yield(i, j)

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

sol = TSP_3D(solve=True)
sol.output()

'''
from datetime import datetime, timedelta
iters = 5
avgTime = timedelta(0)
avgDistance = 0
minDist = 500000000000
for i in range(iters):
	startTime = datetime.now()
	solution = TSP_3D(solve=True)
	path = solution.output()
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
#children = sol.crossover([[1,2,3,4,5,6,7,8,9],[7,4,1,9,2,5,3,6,8]], 3, 5, 4)
#for child in children:
#	print(child)
'''
