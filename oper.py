"""
Authors: Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez,
Héctor Gibrán Ceballos-Cancino, and Francisco Javier Cantu-Ortiz.
Year: 2021.
https://github.com/benjaminsainz/f1-ecac
"""

import numpy as np


def binary_tournament(pop):
    pop_size = len(pop)
    i, j = np.random.randint(pop_size), np.random.randint(pop_size)
    while j == i:
        j = np.random.randint(pop_size)
    if pop[i]['fitness'] > pop[j]['fitness']:
        return pop[i]
    else:
        return pop[j]


def crossover(parent_1, parent_2, rate, n_clusters):
    if np.random.random() >= rate:
        return parent_1
    k_set = list(range(n_clusters))
    flag = False
    child = parent_1
    while flag is False:
        point = np.random.randint(len(parent_1))
        child = parent_1[:point] + parent_2[point:]
        flag = True
        for k in k_set: 
            if k not in child:
                flag = False
    return child


def point_mutation(ind, rate, n_clusters):
    if np.random.random() >= rate:
        return ind
    k_set = list(range(n_clusters))
    flag = False
    child = ind.copy()
    while flag is False:
        for _ in range(int(len(ind) * 0.05)):
            j = np.random.randint(len(ind)-1)
            child[j] = ind[j+1]
        flag = True
        for k in k_set:
            if k not in child:
                flag = False
                child = ind.copy()
    return child


def reproduce(selected, pop_size, p_cross, p_mutation, n_clusters):
    children = []
    for i, parent_1 in enumerate(selected):
        parent_2 = selected[i+1] if i % 2 == 0 else selected[i-1]
        if i == len(selected)-1:
            parent_2 = selected[0]
        child = dict()
        child['partition'] = crossover(parent_1['partition'], parent_2['partition'], p_cross, n_clusters)
        child['partition'] = point_mutation(child['partition'], p_mutation, n_clusters)
        children.append(child)
        if len(children) >= pop_size:
            break
    return children
