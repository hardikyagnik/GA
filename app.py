import numpy as np
import matplotlib.pyplot as plt
import ga

class ProblemStruct():
    def __init__(self, costFunc, nVar, varMin, varMax):
        self.costFunc = costFunc
        self.nVar = nVar
        self.varMin = varMin
        self.varMax = varMax

class ParamsStruct():
    def __init__(self, maxit, npop, pc, gamma, mu, sigma):
        self.maxit = maxit
        self.npop = npop
        self.pc = pc
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma

# Sphere Test Function 
def sphere(x):
    return sum(x**2)

# Problem Definition
problem = ProblemStruct(
    costFunc = sphere, 
    nVar = 5, 
    varMin = -10,
    varMax = 10
)

# GA Parameters
params = ParamsStruct(
    maxit = 200,
    npop = 50,
    pc=1,
    gamma=0.1,
    mu = 0.01,
    sigma = 0.1
)

# Run GA
out = ga.run(problem, params)
print(0)

# Results
# plt.plot(out.bestcost)
plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()