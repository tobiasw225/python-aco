# python-aco
__Author__:Tobias Wenzel

__Name__: PSO Implementation and P-ACO Implementation


## Ant Colony Optimizer (ACO)
Implementation of the ACO with the possibility to visualise error, shortest path and the pheromone-matrix. This is done in the following way: 

Each ant guesses a paths through all points/cities according to a heuristic and the current pheromone value on the connection between the corresponding points. Meaning: The more pheromone and the closer, the higher the probability of choosing the city as next position in the path. After an iteration, the lenghts of the paths are compared and the connections between the cities of the winning ant get all of
the pheromone (could be done otherwise). Also, each connection is
decreased by (1-gamma). This is done until i==num_runs.


## Population based Ant Colony Optimizer (P-ACO)

Simple implementation of the P-ACO algorithm. This is similar to ACO, but there is no evaporation step. In this case a population of solution influences the choice of the ants. After 'population_size' steps, the solution looses it's impact and the corresponding pheromone value is removed from the pheromone matrix.

__Purpose__: Fun + Prep. for exam.

## relevant literature

- Application to Shortest Path Problems: [Dorigo et al.,91] 
- Population based ACO (P-ACO): [Guntsch,Middendorf,2002]