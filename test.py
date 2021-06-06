"""
@authors: Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez,
Héctor Gibrán Ceballos-Cancino, Francisco Javier Cantu-Ortiz
"""

from gen import *
from retr import *


full = ['aggregation', 'breast-cancer-wisconsin', 'breast-tissue', 'dermatology', 'ecoli', 'forest', 'glass', 'iris',
       'jain', 'leaf', 'liver', 'parkinsons', 'pathbased', 'r15', 'seeds', 'segment', 'spiral', 'transfusion', 'wine',
       'zoo']


def test(ds=full, pop_size=200, max_gens=200, runs=1):
    for d in ds:
        data, n_clusters, X, y = retrieval(d)
        f1ecac_run(X, n_clusters, data, pop_size, max_gens, runs=runs, y=y, log_file=False, evolutionary_plot=False)

        
test(full)
