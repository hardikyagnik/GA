import copy
import numpy as np

class IndividualStruct():
    def __init__(self, position, cost):
        self.position = position
        self.cost = cost
    
    def repeat(self, size=0):
        arr = list()
        for i in range(size):
            arr.append(copy.deepcopy(self))
        return arr

class OutputStructure():
    def __init__(self, population, bestsol, bestcost):
        self.population = population
        self.bestsol = bestsol
        self.bestcost = bestcost
    
def run(problem, params):
    # Problem Information
    costFunc = problem.costFunc
    nVar = problem.nVar
    varMin = problem.varMin
    varMax = problem.varMax

    # Parameters 
    maxit = params.maxit
    npop = params.npop
    pc = params.pc
    numChildren = int(np.round(pc*npop/2)*2)
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma

    # Empty Individual Template
    empty_individual = IndividualStruct(
        position=None, 
        cost=None
    )

    # Best Solution Ever Found
    bestsol = copy.deepcopy(empty_individual)
    bestsol.cost = np.inf

    # Intialise Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        pop[i].position = np.random.uniform(varMin, varMax, nVar)
        pop[i].cost = costFunc(pop[i].position)
        if pop[i].cost < bestsol.cost:
            bestsol = copy.deepcopy(pop[i])

    # Best Cost of Iterations
    bestcost = np.empty(maxit)

    # Main Loop
    for it in range(maxit):
        popc = []
        for k in range(numChildren//2):

            # Select Parents
            q = np.random.permutation(npop)
            p1 = pop[q[0]]
            p2 = pop[q[1]]
            
            # Perform Crossover
            c1, c2 = crossover(p1, p2, gamma)

            # Perform Mutation            
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

            # Apply Bound
            apply_bound(c1, varMin, varMax)
            apply_bound(c1, varMin, varMax)

            # Evaluate First Offspring
            c1.cost = costFunc(c1.position)            
            if c1.cost < bestsol.cost:
                bestsol = copy.deepcopy(c1)
            
            c2.cost = costFunc(c2.position)            
            if c2.cost < bestsol.cost:
                bestsol = copy.deepcopy(c2)

            # Add Offspring to popc
            popc.append(c1)
            popc.append(c2)

        # Merge, Sort and Select
        pop += popc
        sorted(pop, key=lambda x: x.cost)
        pop = pop[0:npop]

        # Store Best Cost
        bestcost[it] = bestsol.cost

        # Show Iteration Information
        print(f"Iteration {it}: Best Cost = {bestcost[it]}")

    # Output
    out = OutputStructure(
        population=pop,
        bestsol= bestsol,
        bestcost = bestcost
    )
    return out

def crossover(p1, p2, gamma=0.1):
    c1 = copy.deepcopy(p1)
    c2 = copy.deepcopy(p2)

    alpha = np.random.uniform(-gamma, 1+gamma, *c1.position.shape)
    c1.position = alpha*p1.position + (1-alpha)*p2.position
    c2.position = alpha*p2.position + (1-alpha)*p1.position
    return c1, c2

def mutate(x, mu, sigma):
    y = copy.deepcopy(x)
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma*np.random.randn(*ind.shape)
    return y

def apply_bound(x, varMin, varMax):
    x.position = np.maximum(x.position, varMin)
    x.position = np.minimum(x.position, varMax)