# F1-ECAC

**Authors:** Benjamin Mario Sainz-Tinajero, Andres Eduardo Gutierrez-Rodriguez, Hector G. Ceballos, Francisco J. Cantu-Ortiz  
**Paper title:** F1-ECAC: Enhanced Evolutionary Clustering Using an Ensemble of Supervised Classifiers

Source code of F1-ECAC, an enhanced clustering algorithm that uses supervised learning for solving an unsupervised learning problem designed following the pipeline of the genetic algorithm. The algorithm starts by generating a random initial population that goes through a reproductive process through crossover and mutation operators to create high-quality partitions using the principle of generalization as the clustering criterion to be optimized. F1-ECAC's evolutionary process relies on the ensemble of classifiers set in the objective function to evaluate partitions without any bias, making it an appropriate solution for clustering data in real-world data mining tasks without apriori knowledge of the cluster structure.

F1-ECAC is available in this repository in a Python implementation.

# Algorithm hyper-parameters
``X``: an array containing the dataset features with no header. Each row must belong to one object with one column per feature.  
``n_clusters``: int with the number of desired clusters.  
``data``: a string with the name of the dataset used for printing the algorithm initialization and naming the output file.  
``pop_size`` (default = 200): population size that is carried along the evolutionary process.   
``max_gens`` (default = 200): maximum generations in the evolutionary process.   
``p_crossover`` (default = 0.95): probability of running the crossover operator.  
``p_mutation`` (default = 0.98): probability of running the mutation operator.  
``runs`` (default = 10): independent runs of the algorithm.  
``y`` (default = None): one-dimensional array with the ground truth cluster labels if available.  
``log_file`` (default = False): creates a .csv file with the fitness value of the best individual per generation.  
``evolutionary_plot`` (default = False): creates multiple .jpg files with scatter plots of the first two columns from the dataset and their cluster membership.  

### Optional data retrieval function
An additional data retrieval function is included for easy access and generation of the parameters X, clusters and data along with multiple datasets ready to be clustered, which can be used as a reference for preparing your data. The function will use the datasets included in the path ``/data`` and returns the data string, the X features, and the dataset's number of reference classes (n_clusters). The only parameter for this function is a string with a dataset name from the options. To run it on Python and get the information of the *wine* dataset, run these commands in the interface.     
``>>> from retr import *``  
``>>> data, n_clusters, X, y = data_retrieval('wine')``  

Label files are included for every dataset for any desired benchmarking tests.

# Setup and run using Python
Open your preferred Python interface and follow these commands to generate a clustering using F1-ECAC. To execute it, just import the functions in *gen.py* and run ``f1ecac_run()`` with all of its parameters. See the example code below, which follows the data, n_clusters, X, and y variables set previously for the *wine* dataset.  
**Important**: You will need to have previously installed some basic data science packages such as numpy, pandas, matplotlib, seaborn, and Sci-kit Learn).

``>>> from gen import *``  
``>>> f1ecac_run(X, n_clusters, data, pop_size=200, max_gens=200, p_crossover=0.95, p_mutation=0.98, runs=10, y=y, log_file=True, evolutionary_plot=True)``  

Running these commands will execute F1-ECAC using the wine dataset's features, 3 clusters, 200 individuals per population, 200 generations, probabilities of running the crossover and mutation operators of 0.95 and 0.98 for 10 independent runs, and will compute the adjusted RAND index between the solutions and the provided y array. A .csv file with the clustering and the results is stored in the ``/f1-ecac-out`` path.

A test.py file is provided for a more straight-forward approach to using the algorithm.  

I hope F1-ECAC is a useful tool for your data mining tasks,

Benjamin  
**Email:** a01362640@itesm.mx, bm.sainz@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/benjaminmariosainztinajero/

# References
[1] B. M. Sainz-Tinajero, A. E. Gutierrez-Rodriguez, H. G. Ceballos and F. J. Cantu-Ortiz, "F1-ECAC: Enhanced Evolutionary Clustering Using an Ensemble of Supervised Classifiers," in IEEE Access, 2021, DOI: 10.1109/ACCESS.2021.3116092.
