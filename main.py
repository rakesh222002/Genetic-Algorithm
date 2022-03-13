import numpy as np
import random
from client import get_errors, submit
from prettytable import PrettyTable 
SECRET_KEY = 'WaFuAiyFsluS298o9Cev5zzT3LfpI4Rk5dB0MEXsjemYOfBqK2'
features = 11
pop_length = 12
limit = 10
generations = 10
prob = 0.2

def mutatenoind(vector):
    ind = random.randint(0,features-1)
    if random.uniform(0,1)<0.5:
        vector[ind] = random.uniform(-limit, limit)
    return vector


def generate_individual(individual):
    for i in range(len(individual)):
        if random.uniform(0,1)<=0.5:
            individual[i] = random.uniform(-limit, limit)
    return individual

def crossover(x,y):
    #uniform crossover
    x_mod = []
    y_mod = []
    for i in range(0,len(x)):
        if random.uniform(0,1)<=0.5:
            x_mod.append(y[i])
            y_mod.append(x[i])
        else:
            x_mod.append(x[i])
            y_mod.append(y[i])
    return x_mod, y_mod

if __name__ == "__main__":
    init_vec = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]
    #parent properties
    min_err = 0
    res_vec = np.zeros(pop_length)
    send_vec = []

    ####
    table_parent = PrettyTable(["Parent_Index", "Parents"])
    table_child = PrettyTable(["Child_Index", "Child", "Selected"])
    table_mutated = PrettyTable(["Mut_Index", "Mutated_Child"])
    ####
    
    parent_population = np.zeros((pop_length, features))
    parent_errors = np.zeros(pop_length)
    parent_ve = np.zeros(pop_length)
    parent_te = np.zeros(pop_length)
    parent_probablities = np.zeros(pop_length)
    temp_array = []
    #getting intial population by mutating
    print("Parent")
    for i in range(pop_length):
        temp_parent = np.copy(init_vec)
        parent_population[i] = generate_individual(temp_parent)
        temp_array.append(i)
        temp_array.append(parent_population[i])
        print(i,".", end="")
        print(temp_array[1])
        table_parent.add_row(temp_array)
        temp_array.clear()
    print(table_parent)
    table_parent.clear_rows()
    #getting errors of initial population
    for i in range(pop_length):
        temp_parent = parent_population[i].tolist()
        errors = get_errors(SECRET_KEY, temp_parent)
        parent_errors[i] = errors[0] + errors[1]
        parent_te[i] = errors[0]
        parent_ve[i] = errors[1]
    
    #creating generations
    for gen in range(generations):
        #sorting based on errors
        sort_indices = parent_errors.argsort()
        parent_errors = parent_errors[sort_indices[::1]]
        parent_te = parent_te[sort_indices[::1]]
        parent_ve = parent_ve[sort_indices[::1]]
        parent_population = parent_population[sort_indices[::1]]
        #probablity generation
        
        for k in range(0,pop_length-1):
            parent_probablities[k]=((1-prob)**k)*prob
        parent_probablities[pop_length-1] = (1-prob)**(pop_length-1)
        
        #child properties
        child_population = np.zeros((pop_length, features))
        child_errors = np.zeros(pop_length)
        child_te = np.zeros(pop_length)
        child_ve = np.zeros(pop_length)
        print("After Crossover")
        for k in range(int(pop_length/2)):
            selected = np.random.choice(np.arange(0, pop_length), 2, replace=False, p=parent_probablities)
            child_population[2*k], child_population[(2*k)+1] = crossover(parent_population[selected[0]], parent_population[selected[1]])
            temp_array.append(2*k)
            temp_array.append(child_population[2*k])
            temp_array.append(selected)
            print(2*k,".", end="")
            print(temp_array[1])
            table_child.add_row(temp_array)
            temp_array.clear()
            temp_array.append((2*k)+1)
            temp_array.append(child_population[(2*k)+1])
            temp_array.append(selected)
            print(2*k + 1,".", end="")
            print(temp_array[1])
            table_child.add_row(temp_array)
            temp_array.clear()
        child_population[pop_length-1].fill(0)
        print(table_child)
        table_child.clear_rows()
        print("After Mutation")
        for k in range(pop_length):
            temp_child = np.copy(child_population[k])
            child_population[k] = mutatenoind(temp_child)
            temp_array.append(k)
            temp_array.append(child_population[k])
            print(k,".", end="")
            print(temp_array[1])
            table_mutated.add_row(temp_array)
            temp_array.clear()
        print(table_mutated)
        table_mutated.clear_rows()
        print("errors")
        for k in range(pop_length):
            temp_child = child_population[k].tolist()
            errors = get_errors(SECRET_KEY, temp_child)
            child_errors[k] = errors[0] + errors[1]
            child_te[k] = errors[0]
            child_ve[k] = errors[1]
            print("Generation : ", end="")
            print(gen+1)
            print("Vector : ", end ="")
            print(child_population[k])
            print("error : ",end="")
            print(child_errors[k])

        #obtaining best vector

        for k in range(pop_length):
            if min_err==0 or min_err > child_errors[k]:
                min_err = child_errors[k]
                res_vec = child_population[k]
            send_vec.append((child_population[k], child_errors[k]))
            
        #children to parent
        print("Generation : ", gen+1)
        print("parent")
        for k in range(pop_length):
            parent_population[k] = child_population[k]
            parent_errors[k] = child_errors[k]
            parent_te[k] = child_te[k]
            parent_ve[k] = child_ve[k]
            temp_array.append(k)
            temp_array.append(parent_population[k])
            print(k,".", end="")
            print(temp_array[1])
            table_parent.add_row(temp_array)
            temp_array.clear()
        print(table_parent)
        table_parent.clear_rows()

    temp_res = res_vec.tolist() 
    submitflag = submit(SECRET_KEY, temp_res)
    print(submitflag)
    print("Vector submitted : ", end="")
    print(res_vec)
    print("error : ", end="")
    print(min_err)
    print("Send : ")
    send_vec.sort(key = lambda x: x[1])
    for i in range(len(send_vec)):
        print("Vector :")
        print(send_vec[i][0])
        print("Error :")
        print(send_vec[i][1])