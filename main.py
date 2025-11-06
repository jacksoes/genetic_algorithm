import random
import numpy as np
import heapq


# 1. create initial population
# 2. create fitness function
# 3. SELECTION 4. CROSSOVER, 5. MUTATION
# 6. repeate steps 3, 4 and 5 

class genetic_ordering:
    #states: fitnessScore, cost
    matrix = [[]]
    cost = 0
    fitness_score = 0.00
    pair = None
    
    def __init__(self, matrix, cost):
        self.matrix = matrix
        self.cost = cost
    def __lt__(self, other):
        return self.fitness_score < other.fitness_score    






def create_random_population(rows, cols):
    # Define the total number of unique values you need
   
    num_elements = rows * cols

    # Choose unique random numbers (no duplicates)
    random_int_array = np.random.choice(np.arange(1, (rows * cols) + 1), size=num_elements, replace=False)

    # Reshape to 2D
    random_int_array = random_int_array.reshape(rows, cols)

    #print(random_int_array)
    return random_int_array

def calculate_cost(arr):
    #numbers should be in order left to right
    #numbers should be in correct order from top to bottom
    
    #must loop through every item in arr, solution will be O(N).
 
    
    #loop through array in order and keep track of count
    # count starting at 1 is the correct number, abs(correct_number - actual) represents difference between,
       
    cost = 0
    correct_number = 1
    for i in arr:
        for j in i:
            #print(j)
            #print("correct number ", correct_number)
            cost += abs(correct_number - j)
            correct_number += 1
    
    
    #print(cost)
    return cost   
'''
def looop(population):
    
    for i in range(len(population)):
        print("CALCULATED COOOOST", calculate_cost(population[i].matrix))
        

    
    total_cost = 0
    for i in range(len(population)):
        total_cost += population[i].cost
        
    
    max_heap = []
    for i in range(len(population)):
        population[i].fitness_score = (total_cost - population[i].cost) / total_cost
        heapq.heappush(max_heap, (-population[i].fitness_score, population[i]))
        
        
        

    #print(f"Heap after pushes: {max_heap}")
    new_population = {}
    
    limit = 0
    if ((len(population) // 2) % 2 == 0):
        limit = len(population) // 2
    else:
        limit = (len(population) // 2) - 1
        
    for i in range(limit):    
        score, obj = heapq.heappop(max_heap)
        new_population[i] = obj
    
    #for i in range(len(new_population)):    
        #print("NEW POPULATION", new_population[i].matrix )
    
    #for i in range(len(new_population) - 1):
        
     #   mid = len(new_population[i].matrix) // 2

        # Slice row-wise
      #  top_half = new_population[i].matrix[:mid].copy()      # first half of the rows
       # bottom_half = new_population[i+1].matrix[mid:].copy()   # second half of the rows
        # may be able to get rid of a .copy for optimization
        
        #new_population[i].matrix[:mid] = bottom_half 
        
        #new_population[i + 1].matrix[mid:] = top_half
    for i in range(0, len(new_population) - 1, 2):  # step by 2 → (0,1), (2,3), (4,5)...
        parent1 = new_population[i].matrix.copy()
        parent2 = new_population[i + 1].matrix.copy()

        rows = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Randomly swap rows between the two
        for r in range(rows):
            if random.random() < 0.5:  # 50% chance to swap this row
                child1[r, :], child2[r, :] = parent2[r, :], parent1[r, :]

        # Assign new children back into population
        new_population[i].matrix = child1
        new_population[i + 1].matrix = child2
        
    
    for i in range(len(new_population)):    
        #print("NEWEST POPULATION", new_population[i].matrix )
        score = calculate_cost(new_population[i].matrix)
        #print(score)
    for i in range(len(population) // 2):
        new_population[len(new_population)] = population[i]
    for i in range(len(new_population)):
        rows, cols = new_population[i].matrix.shape
        r1, c1 = random.randrange(rows), random.randrange(cols)
        r2, c2 = random.randrange(rows), random.randrange(cols)

        # Swap the two values
        new_population[i].matrix[r1, c1], new_population[i].matrix[r2, c2] = new_population[i].matrix[r2, c2], new_population[i].matrix[r1, c1]
       
    print("pop set")    
    for i in range(len(new_population)):
        print(new_population[i].matrix)
    return calculate_cost(population[i] == 0)
'''    
    
    
    # ADD EVEN/ODD NUMBERS TO HEAP AND SWAP INDEX  1ST GENOME WILL HAVE MINHEAP AND TOP HALF, 2ND WILL HAVE MAXHEAP AND BOTTOM HALF
    

def crossover(arr1, arr2, even_swap):
    #loop matrix
    # if odd store index in dictionary with num
    arr1_swap = {}
    arr2_swap = {}       
    
    newarr1 = arr1.copy()
    newarr2 = arr2.copy()
    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            if(even_swap % 2 == 0):
                if(arr1[i][j] % 2 == 0):
                    arr1_swap[str(i) + str(j)] =  arr1[i][j]
                if(arr2[i][j] % 2 == 0):
                    arr2_swap[str(i) + str(j)] = arr2[i][j]
            else:
                if(arr1[i][j] % 2 == 1):
                    arr1_swap[str(i) + str(j)] =  arr1[i][j]
                if(arr2[i][j] % 2 == 1):
                    arr2_swap[str(i) + str(j)] = arr2[i][j]
                
                
    arr1_swap_shuffled = list(arr1_swap.items())
    random.shuffle(arr1_swap_shuffled)
    arr1_swap_new = dict(arr1_swap_shuffled)
    
    arr2_swap_shuffled = list(arr2_swap.items())
    random.shuffle(arr2_swap_shuffled)
    arr2_swap_new = dict(arr2_swap_shuffled)
                
    for index1, index2 in zip(arr1_swap_new, arr2_swap_new):
        row1 = int(index1[0]) 
        col1 = int(index1[1])        
        row2 = int(index2[0]) 
        col2 = int(index2[1])
        
        #print(arr2_swap[index2])
        #if(( arr2_swap_new[index2] < newarr1[row1][col1]) and ((row1 >= row2) and (col1 >= col2)) ):
        
    if newarr1[row1][col1] != row1 * 5 + col1 + 1:
        newarr1[row1][col1] = arr2_swap_new[index2]

    if newarr2[row2][col2] != row2 * 5 + col2 + 1:
        newarr2[row2][col2] = arr1_swap_new[index1]
        
    if(calculate_cost(newarr1) < calculate_cost(arr1)):
        arr1[:, :] = newarr1  
    if(calculate_cost(newarr2) < calculate_cost(arr2)):
        arr2[:, :] = newarr2 
        #print(arr1)
        #print(arr2)
            
    #for index in arr2_swap:
     #   row = int(index[0]) 
      #  col = int(index[1])
       # arr1[row][col] = arr2_swap[index]
    
    #SPLIT INITIAL ARRAY FIRST????
    # get all even/odd numbers from arr1 into min heap
    # get all even/odd numbers from arr2 into max heap
    # with index too
    #swap by index starting only if maxheaphead.index < mineheaphead.index
def mutation(arr, first, sec):
    
    temp = arr[first[0]][first[1]]
    arr[first[0]][first[1]] = arr[sec[0]][sec[1]]
    arr[sec[0]][sec[1]] = temp
    
    
    pass
    #loop through matrix monolithic stack till inorder is detected.
    #start with first num in matrix, iterate until     
     

def main():
    # create dictionary of initial population
    rows = int(input("M = "))
    cols = int(input("N = "))
    population = []
    for i in range(100):
        arr = create_random_population(rows, cols)
        cost = calculate_cost(arr)
        population.append(genetic_ordering(arr, cost))
        
    swap_turn = 0   
    
    test = 0
    count = 0 
    while count < len(population) - 1:
        #print(calculate_cost(population[count].matrix))
        #print(calculate_cost(population[count + 1].matrix))
        
        i1 = random.randint(0, len(population) - 1)
        i2 = random.randint(0, len(population) - 1)
        while i2 == i1:
            i2 = random.randint(0, len(population) - 1)
        
        population = sorted(population, key=lambda x: x.cost)
        print("lowest cost: ", population[0].cost)

        top_individuals = population[:len(population)//2]
        
        random.shuffle(top_individuals)
        for i in range(0, len(top_individuals) - 1, 2):
            crossover(top_individuals[i].matrix, top_individuals[i+1].matrix, swap_turn)
            swap_turn += 1
            
            
            
            mutation_row = random.randint(0, rows -1)
            mutation_col = random.randint(0, cols -1)
            
            mutation_row2 = random.randint(0, rows -1)
            mutation_col2 = random.randint(0, cols -1)
            
            first_index = [mutation_row, mutation_col]
            sec_index = [mutation_row2, mutation_col2]
            mutation(top_individuals[i].matrix, first_index, sec_index)
            
            
            mutation_row = random.randint(0, rows -1)
            mutation_col = random.randint(0, cols -1)
            
            mutation_row2 = random.randint(0, rows -1)
            mutation_col2 = random.randint(0, cols -1)
            
            first_index = [mutation_row, mutation_col]
            sec_index = [mutation_row2, mutation_col2]
            mutation(top_individuals[i+1].matrix, first_index, sec_index)
        #crossover(population[i1].matrix, population[i2].matrix, swap_turn)
        #swap_turn += 1
        
       
        
        
        population[i1].cost = calculate_cost(population[i1].matrix)
        population[i2].cost = calculate_cost(population[i2].matrix)   
                
        if(calculate_cost(population[count].matrix) == 0 or calculate_cost(population[count + 1].matrix) == 0):
            break
        
        count += 2
        
        if(count >= len(population)- 1):
            count = 0
        test += 1
        if test % 5 == 0:
            population = sorted(population, key=lambda x: x.cost)[:50]
            for i in range(50):
                arr = create_random_population(rows, cols)
                cost = calculate_cost(arr)
                population.append(genetic_ordering(arr, cost))
   
    print(population[count].matrix)
    print(population[count + 1])
    '''
    new_population = {}
    index = 0
    for i in range(0, len(population) - 1, 2):
        parent1 = population[i]
        parent2 = population[i + 1]

        # Perform crossover between their matrices
        child1_matrix, child2_matrix = crossover(parent1.matrix, parent2.matrix)

        # Recalculate cost (or fitness)
        cost1 = calculate_cost(child1_matrix)
        cost2 = calculate_cost(child2_matrix)

        # Create new genetic_ordering objects for children
        child1 = genetic_ordering(child1_matrix, cost1)
        child2 = genetic_ordering(child2_matrix, cost2)

        # Store them in the new population
        new_population[index] = child1
        new_population[index + 1] = child2
        index += 2

    # Replace or combine old population with new
    population = new_population
    '''  
        
    #while(calculate_cost(population[i].matrix) != 0):
        
    #final_ans = looop(population)
    
    #print("THE FINAL ANS IS: ", final_ans)
    
        
        
        
        
         

    
    
main()