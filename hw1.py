import random

class TSP_3D:
	def __init__(self):
		self.cityCount = None # Assigned by readInput
		self.cities = self.readInput()	

	def readInput(self) -> list[tuple[int, int, int]]:
		cities = []
		with open('input.txt', 'r') as file:
			self.cityCount = file.readline()
			for city in file.readlines():
				cities.append(tuple(city.strip().split()))
			
		return cities

	def createInitialPopulation(self, size: int, cities: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
		initial_population = cities
		random.shuffle(initial_population)
		return initial_population

	def createMatingPool(self, population: list[tuple[int, int, int]], rankList: list[tuple[int, int]]) -> list:
		pass

solution = TSP_3D()
print(solution.cities, solution.cityCount)
print(solution.createInitialPopulation(
