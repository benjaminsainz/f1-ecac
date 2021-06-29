"""
Authors: Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez,
Héctor Gibrán Ceballos-Cancino, and Francisco Javier Cantu-Ortiz.
Year: 2021.
https://github.com/benjaminsainz/f1-ecac
"""

from ind import *
from obj import *
from oper import *
import numpy as np
import pandas as pd
import os
import glob
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.gridspec as gridspec


def f1ecac_run(X, n_clusters, data, pop_size=200, max_gens=200, p_crossover=0.95, p_mutation=0.98, runs=10, y=None,
               log_file=False, evolutionary_plot=False):
    genetic_palette = sns.cubehelix_palette(n_clusters, start=0.5, rot=-0.75)
    for run in range(runs):
        print('============= TEST {} ============='.format(run + 1))
        print('Clustering started using F1-ECAC'.format(data))
        print('Dataset: {}, Clusters: {}, Instances: {}, Features: {}'.format(data, n_clusters, len(X), len(X[0])))
        print('Population size: {}, Generations: {}'.format(pop_size, max_gens))

        start = time.time()
        population = []
        best_fit_log = []
        avg_fit_log = []
        X = StandardScaler().fit_transform(X)

        print('Generating initial population')
        for _ in range(pop_size):
            individual = {'partition': random_gen(n_clusters, X)}
            individual['fitness'] = fitness_value(X, individual['partition'])
            if individual not in population:
                population.append(individual)
        best = sorted(population, key=lambda k: k['fitness'], reverse=True)[0]

        print('Starting genetic process...')
        for i in range(max_gens):
            print('Generation {}, Fitness: {:.4f}, Elapsed Time: {:.2f}s'.format(i + 1, best['fitness'],
                                                                                 time.time() - start))
            selected = []
            for _ in range(pop_size):
                selected.append(binary_tournament(population))
            children = reproduce(selected, pop_size, p_crossover, p_mutation, n_clusters)
            for j in range(len(children)):
                children[j]['fitness'] = fitness_value(X, children[j]['partition'])
            children.sort(key=lambda l: l['fitness'], reverse=True)
            if children[0]['fitness'] >= best['fitness']:
                best = children[0]
            population = children
            best_fit_log.append((i + 1, best['fitness']))
            if log_file:
                avg_fitness_gen = []
                for child in children:
                    avg_fitness_gen.append(child['fitness'])
                    avg_fit_log.append((i + 1, np.mean(avg_fitness_gen)))
            if evolutionary_plot:
                sns.set_theme()
                sns.set_style("dark")
                fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
                plot_df = pd.DataFrame(X)
                plot_df['y'] = list(best['partition'])
                for q in range(n_clusters):
                    sns.scatterplot(data=plot_df[plot_df['y'] == q], x=0, y=1, marker='D', ax=ax, color=genetic_palette[q])
                plt.title('F1-ECAC - Generation {}'.format(i+1))
                plt.tight_layout()
                if not os.path.exists('figures/{}/{}'.format(data, run+1)):
                    os.makedirs('figures/{}/{}'.format(data, run+1))
                plt.savefig('figures/{}/{}/scatter_{}.jpg'.format(data, run+1, i+1), format='jpg')
            if best['fitness'] == 1:
                break

        run_time = time.time() - start
        best['time'] = run_time
        print('Optimization finished in {:.2f}s with an objective of {:.4f}'.format(best['time'], best['fitness']))
        best['partition'] = np.array(best['partition'])
        if y is not None:
            adj_rand_index = adjusted_rand_score(y, best['partition'])
            print('Adjusted RAND index: {:.4f}'.format(adj_rand_index))
        else:
            adj_rand_index = np.nan
            print('No labels provided')

        d = dict()
        d['Dataset'] = data
        d['Algorithm'] = 'f1-ecac'
        d['Clusters'] = n_clusters
        d['Instances'] = len(X)
        d['Features'] = len(X[0])
        d['Pop. size'] = pop_size
        d['Max. gens'] = max_gens
        d['No. objectives'] = 1
        d['Obj. 1 name'] = 'f1'
        d['Objective 1'] = best['fitness']
        d['Obj. 2 name'] = np.nan
        d['Objective 2'] = np.nan
        d['Time'] = best['time']
        d['Adjusted Rand Index'] = adj_rand_index
        for i in range(len(best['partition'])):
            d['X{}'.format(i + 1)] = '{}'.format(best['partition'][i])

        out = pd.DataFrame(d, index=[data])
        if not os.path.exists('f1-ecac-out/{}_{}_{}_{}'.format(data, n_clusters, pop_size, max_gens)):
            os.makedirs('f1-ecac-out/{}_{}_{}_{}'.format(data, n_clusters, pop_size, max_gens))
        out.to_csv('f1-ecac-out/{}_{}_{}_{}/solution-{}_{}_{}_{}-{}.csv'.format(data, n_clusters, pop_size, max_gens,
                                                                                data, n_clusters, pop_size, max_gens,
                                                                                run + 1), index=False)
        if log_file:
            best_log = pd.DataFrame(best_fit_log, columns=['gen', 'fitness'])
            best_log.to_csv('f1-ecac-out/{}_{}_{}_{}/best-log-{}_{}_{}_{}-{}.csv'.format(data, n_clusters, pop_size,
                                                                                         max_gens, data, n_clusters,
                                                                                         pop_size, max_gens, run + 1),
                            index=False)
            avg_log = pd.DataFrame(avg_fit_log, columns=['gen', 'fitness'])
            avg_log.to_csv('f1-ecac-out/{}_{}_{}_{}/avg-log-{}_{}_{}_{}-{}.csv'.format(data, n_clusters, pop_size,
                                                                                       max_gens, data, n_clusters,
                                                                                       pop_size, max_gens, run + 1),
                           index=False)

        filenames = glob.glob('f1-ecac-out/{}_{}_{}_{}/solution*'.format(data, n_clusters, pop_size, max_gens))
        df = pd.DataFrame()
        for name in filenames:
            temp_df = pd.read_csv(name)
            df = df.append(temp_df)
        df.reset_index(drop=True, inplace=True)
        df.to_csv('f1-ecac-out/solutions-{}_{}_{}_{}-{}.csv'.format(data, n_clusters,
                                                                    pop_size, max_gens, runs))
