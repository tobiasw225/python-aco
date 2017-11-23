# python-aco
_Author_:Tobias Wenzel

_Name_: PSO Implementation and P-ACO Implementation

_Description_: Implementation of the ACO with the possibility to visualise error,
shortest path and the pheromone-matrix. This is done in the following
way: each ant guesses a paths through all points/cities according to a
heuristic and the current pheromone value on the connection between
the corresponding points. Meaning: The more pheromone and the closer,
the higher the probability of choosing the city as next position in
the path. After an iteration, the lenghts of the paths are compared
and the connections between the cities of the winning ant get all of
the pheromone (could be done otherwise). Also, each connection is
decreased by (1-gamma). This is done until i==num_runs.

Simple implementation of the P-ACO algorithm. This is similar to ACO,
but there is no evaporation step. In this case a population of
solution influences the choice of the ants. After 'population_size'
steps, the solution looses it's impact and the corresponding pheromone
value is removed from the pheromone matrix.