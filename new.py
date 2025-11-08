import random
import numpy as np



class Genome:
    #states: fitnessScore, cost, matrix
    matrix = [[]]
    cost = 0
    fitness_score = 0.00
    
    def __init__(self, matrix):
        self.matrix = matrix
        self.cost = calculate_cost(self.matrix)
    def __lt__(self, other):
        return self.fitness_score < other.fitness_score  
    
    def update_cost(self):
        self.cost = calculate_cost(self.matrix)

    def update_matrix(self, matrix):
        self.matrix = matrix
        self.update_cost()

def calculate_cost(arr):
    cost = 0
    correct_number = 1
    for i in arr:
        for j in i:
            cost += abs(correct_number - j)
            correct_number += 1
    return cost   


def fix_duplicates(arr):
    # Flatten the array
    flat = [x for row in arr for x in row]
    all_vals = set(flat)

    # create a set of numbers 1 through n
    expected = set(range(1, len(flat) + 1))

    # find amount of missing values
    missing = list(expected - all_vals)
    seen = set()
    i_missing = 0

    # replace duplicate numbers with missing
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] in seen:
                arr[i][j] = missing[i_missing]
                i_missing += 1
            seen.add(arr[i][j])

    return arr





def crossover(arr1, arr2):
   

    rows = len(arr1)
    cols = len(arr1[0])

    child = [row[:] for row in arr1]  # start with a copy of arr1

    # crossover rows
    if random.random() < 0.9:
        mid = random.randint(1, rows - 1)
        child[:mid] = arr1[:mid]
        child[mid:] = arr2[mid:]

    # crossover columns
    if random.random() > 0.1:
        mid = random.randint(1, cols - 1)
        child = [r1[:mid] + r2[mid:] for r1, r2 in zip(child, arr2)]

    child = fix_duplicates(child)
    return child


def mutate(arr, bias):

    
    rows = len(arr)
    cols = len(arr[0])


    r1, c1 = random.randint(0, rows - 1), random.randint(0, cols - 1)
    r2, c2 = random.randint(0, rows - 1), random.randint(0, cols - 1)

    val1 = arr[r1][c1]
    val2 = arr[r2][c2]

    
    
    rand = random.random()
    

    if(bias > rand):
        if val1 > val2 and (r1 < r2 or (r1 == r2 and c1 < c2)):
            arr[r1][c1], arr[r2][c2] = arr[r2][c2], arr[r1][c1]
    else:
        arr[r1][c1], arr[r2][c2] = arr[r2][c2], arr[r1][c1]
    
    return arr
        
  

def create_random_2d_array(rows, cols):
    numbers = list(range(1, rows * cols + 1))
    random.shuffle(numbers)
    arr = [numbers[i*cols:(i+1)*cols] for i in range(rows)]
    return arr











def weighted_selection(population, k):

    #costs for population
    costs = [genome.cost for genome in population]


    # assign weights a probabilities
    max_cost = max(costs)
    weights = [(max_cost - c + 1e-6) for c in costs] 

    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    selected = random.choices(population, weights=probabilities, k=k)
    return selected





def main():
    n = int(input("n = "))
    m = int(input("m = "))

    initial_population = [] #total = [create_random_2d_array(37, 35) for _ in range(100)]

    for i in range(200):
        matrix = create_random_2d_array(n, m)
        initial_population.append(Genome(matrix))


    
    local_optimum_count = 0
    mutation_count = 1
    best_cost = None
    previous_best_cost = None

    while True:
        #selection
        selection_amount = 20
        selected_parents = weighted_selection(initial_population, selection_amount)


        new_generation = []
        #children_count = len(initial_population) // 2
        for i in range(selection_amount):
            parent1, parent2 = random.sample(selected_parents, 2)
            child_matrix = crossover(parent1.matrix, parent2.matrix)
            child = Genome(child_matrix)

            mutated_matrix = [row[:] for row in child.matrix]
            for i in range(20):
                 mutated_matrix = mutate(mutated_matrix, bias=.75)
            child.update_matrix(mutated_matrix)

            new_generation.append(child)


            
        initial_population.sort(key=lambda g: g.cost)
            

        top_50 = initial_population[:50]

        initial_population = (new_generation + top_50 + initial_population)[:500]
        

      



        initial_population.sort(key=lambda g: g.cost)
            
        best_cost = initial_population[0].cost
        
        if previous_best_cost == initial_population[0].cost:
            local_optimum_count += 1
        else:
            local_optimum_count = 0
            previous_best_cost = initial_population[0].cost
        

        if local_optimum_count >= 10:
            mutation_count += 1
            local_optimum_count = 0
            
        if mutation_count >= 20:
            mutation_count = 1
        
        
        
        print("Best cost:", best_cost)
        if(best_cost == 0):
            print("matrix has been solved", initial_population[0].matrix)
            break

     




            

        



main()



