# python-aco


Implementations of the Ant Colony Optimizer (ACO) to calculate the shortest path for a given Traveling Salesman Problem. 


Consider an ant colony trying to find it's next meal. Each ant has a very limited in it's way of perception and communication. While moving more or less  randomly, they spread pheromone. If an ant finds pheromone it will more likely move into this direction. After a while, most of the ants will choose the same way. The picture below (taken from Wikipedia) illustrates this. 

![ACO illustration](https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Aco_branches.svg/640px-Aco_branches.svg.png)


We will choose this path as our 'optimal' path. 


## Getting Started

Depending on your capabilities and interests you have different options from here:

+ Read some [papers](#Relevant-literature) and dive into the concepts of PSO and H-PSO.
+ Skim over the summaries of [ACO](#ACO) or/ and  [P-ACO](#P-ACO) and have a look at the implementations. Feel free to comment ;)

## Prerequisites

```
pip install numpy
pip install matplotlib
```

## ACO
Implementation of the Ant Colony Optimizer (ACO) with the possibility to visualise error, shortest path and the pheromone-matrix. This is done in the following way: 

Each ant guesses a paths through all points/cities according to a heuristic and the current pheromone value on the connection between the corresponding points. Meaning: The more pheromone and the closer, the higher the probability of choosing the city as next position in the path. After an iteration, the lenghts of the paths are compared and the connections between the cities of the winning ant get all of
the pheromone (could be done otherwise). Also, each connection is
decreased by (1-gamma). 

Extremely simplefied:
```
for i in range(num_runs):
	best_ant = shortest_path()
	update_tau_matrix(best_ant)
	tau_matrix *= (1 - self.gamma) 
```


## P-ACO

Simple implementation of the Population based Ant Colony Optimizer (P-ACO) algorithm. This is similar to ACO, but there is no evaporation step for all all ants. In this case a population of solution influences the choice of the ants. After 'population_size' steps, the solution looses it's impact and the corresponding pheromone value is removed from the pheromone matrix.

Extremely simplefied:
```
for i in range(num_runs):
    best_ant = self.shortest_path()
    self.add_solution(best_ant.current_solution)
```



## Relevant literature

- Application to Shortest Path Problems: [Dorigo et al.,91] 
- Population based ACO (P-ACO): [Guntsch,Middendorf,2002]
