import numpy as np
import random as random
import itertools as itertools
import matplotlib.pyplot as plt
import logging as log
import datetime
import selection from 'roulette_method'

#OPTIMAL SCENARIO
# Had12
# 12
# 1652(OPT)(3, 10, 11, 2, 12, 5, 6, 7, 8, 1, 4, 9)

# Had14
# 14
# 2724(OPT)(8, 13, 10, 5, 12, 11, 2, 14, 3, 6, 7, 1, 9, 4)
#
# Had16
# 16
# 3720(OPT)(9, 4, 16, 1, 7, 8, 6, 14, 15, 11, 12, 10, 5, 3, 2, 13)
#
# Had18
# 18
# 5358(OPT)(8, 15, 16, 6, 7, 18, 14, 11, 1, 10, 12, 5, 3, 13, 2, 17, 9, 4)
#
# Had20
# 20
# 6922(OPT)(8, 15, 16, 14, 19, 6, 7, 17, 1, 12, 10, 11, 5, 20, 2, 3, 4, 9, 18, 13)


#########PROPERTIES##############

ITERATIONS = 100                      #LICZBA ITERACJI
POP_SIZE = 100                      #ROZMIAR POPULACJI
GEN = 150                           #LICZBA POKOLEN
PX = 0.7                             #PRAWDOPODOBIENSTWO KRZYŻOWANIA
PM = 0.2                            #PRAWDOPODOIEŃSTWO MUTACJI
TOUR = 5                            #ROZMIAR TURNIEJU
SAVE_STRONGEST = True              #0 FALSE #1 TRUE
TEST_TWO_SAVE_STRONGEST = False      # TEST TWO VARIATIONS OF SAVE_STORNGEST
SELECTION_METHOD = 1                # 0: ROULETTE 1:TOURNAMENT
TEST_MORE_METHODS = 1               # IF different than 1 SELECTION METHOD should be 0
DATAFILEPATH = 'input/had12.dat'    #PATH TO FILE WITH INITIALISE DATA

########END OF PROPERTIES###########


#load data from file
input_array = np.loadtxt(DATAFILEPATH, skiprows=1)       #load input (two arrays)
dimension = int(open(DATAFILEPATH, 'r').readline())      #read Dimension writed in first row
flow_matrix = input_array[:dimension, :]                 #spit array - first array is FLOW MATRIX
distance_matrix = input_array[dimension:, ]              #split array - second array is DISTANCE_MATRIX

COMBINATIONS = list(itertools.combinations(range(dimension), 2))

def getInitialiseInfo():
    print("Dimension: {}".format(dimension))
    print("Distance matrix: ")
    print(distance_matrix)
    print("Flowmatrix:")
    print(flow_matrix)


def generateIndividual():
    assigments = np.arange(0, dimension)
    random.shuffle(assigments)
    return assigments


def getDistance(assigments, positions):
    a = 0
    try : a = distance_matrix[assigments.tolist().index(positions[0])][assigments.tolist().index(positions[1])]
    except ValueError:
        print("error")
    return a


def getCost(assigments, positions):
    return 2*getDistance(assigments, positions)*getFlow(positions)


def getFlow(positions):
    return flow_matrix[positions[0]][positions[1]]


def costFunction(assigments, combinations):
    cost = 0
    for pair in combinations:
        cost += getCost(assigments, pair)
    return cost


def initialise(populationSize):
    population = np.zeros((populationSize, dimension))
    for i in range(0, populationSize):
        population[i] = generateIndividual()
    return population


def getCostsVector(population):
    costVector = np.zeros(population.shape[0])
    for i in range(0, population.shape[0]):
        costVector[i] = costFunction(population[i], COMBINATIONS)
    return costVector




def getRandomInvidivual():
    return random.randint(0, POP_SIZE-1)


def allUnique(x):
     seen = set()
     return not any(i in seen or seen.add(i) for i in x)


def repair(individual):
    if allUnique(individual):
        return individual
    list = individual.tolist()
    counts = np.arange(0, individual.shape[0], dtype=int)
    lackOf = []
    for i in range(0, individual.shape[0]):
        counts[i] = list.count(i)
        if i not in individual:
            lackOf.append(i)
    for i in range(0, individual.shape[0]):
        while(counts[i] >1 ):
            pop = lackOf.pop()
            list[list.index(i)] = pop
            counts[i] -= 1
            continue
    return np.array(list)


def crossoverDiscrete(firstI, secondI):
    for i in range(0, firstI.shape[0]):
        if(random.uniform(0,1) <0.5):
            firstI[i] = secondI[i]
    firstI = repair(firstI)
    return firstI


def crossover(population):
    bestIndividualIndex = -1
    if(SAVE_STRONGEST == True):
        costVector = getCostsVector(population)
        bestIndividualIndex = costVector.tolist().index(min(costVector))
    for i in range(0, population.shape[0]):
        if(random.uniform(0, 1) < PX and i!=bestIndividualIndex):
            j = getRandomInvidivual()
            population[i] = crossoverDiscrete(population[i], population[j])
    return population


def mutate(individual):
    i = random.randint(0, individual.shape[0] - 1)
    j = random.randint(0, individual.shape[0] - 1)
    individual[i], individual[j] = individual[j], individual[i]
    return individual


def mutation(population):
    bestIndividualIndex = -1
    if(SAVE_STRONGEST == True):
        costVector = getCostsVector(population)
        bestIndividualIndex = costVector.tolist().index(min(costVector))
    for i in range(0, population.shape[0]):
        if (random.uniform(0, 1) < PM and i != bestIndividualIndex):
            mutate(population[i])
    return population


def getBestIndividual(population):
    costVector = getCostsVector(population)
    bestIndividualIndex = costVector.tolist().index(min(costVector))
    return population[bestIndividualIndex]

def preparePlot(field, minOutput, maxOutput, avgOutput, globalMinOutput):
    plt.figure()
    plt.plot(field, minOutput)
    plt.plot(field, maxOutput)
    plt.plot(field, avgOutput)
    plt.plot(field, globalMinOutput)
    plt.suptitle('COST: {}, \n Params: PX: {}, PM: {} GEN: {}'.format("{} - global min 1652".format(min(minOutput)), PX, PM, GEN))
    plt.legend(("min", "max", "avg", "global min"))
    plt.ylabel("Cost")
    if SELECTION_METHOD == 0:
        plt.xlabel("ITERATIONS \n METHOD: ROULETTE SAVE_STRONEGEST: {}".format(SAVE_STRONGEST))
    else:
        plt.xlabel("ITERATIONS \n METHOD: TOURNAMENT TOUR:{} SAVE_STRONEGEST: {}".format(TOUR, SAVE_STRONGEST))


def geneticAlgorithm():
    pop = initialise(POP_SIZE)
    minOutput = []
    avgOutput = []
    maxOutput = []
    globalMinOutput = []
    minOutput.append(min(getCostsVector(pop)))
    maxOutput.append(max(getCostsVector(pop)))
    avgOutput.append(np.average(getCostsVector(pop)))
    globalMinOutput.append(min(minOutput))
    field = np.arange(0, GEN+1)

    for i in range(0, GEN):
        pop = selection(pop)
        pop = crossover(pop)
        pop = mutation(pop)
        minOutput.append(min(getCostsVector(pop)))
        maxOutput.append(max(getCostsVector(pop)))
        avgOutput.append(np.average(getCostsVector(pop)))
        globalMinOutput.append(min(minOutput))

    preparePlot(field, minOutput, maxOutput, avgOutput, globalMinOutput)

    log.info("min valuse {}".format(min(minOutput)))
    print("min value {} - global min 1652".format(min(minOutput)))
    return getBestIndividual(pop)

def getDate(inFile=True):
    if inFile:
        return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def logInfo():
    log.info(getDate())
    if SAVE_STRONGEST == 1:
        log.info("###SAVE_STRONGEST ENABLED")
    else:
        log.info("###SAVE_STRONGEST DISABLED")

    if SELECTION_METHOD ==0:
        log.info("###ROULETTE METHOD")
    else:
        log.info("###TOURNAMENT METHOD")

results=[]
name = 'logs/'+getDate(inFile=False)+'.log'
log.basicConfig(filename=name, level=log.INFO, format='%(message)s')

for i in range(0, 2*TEST_MORE_METHODS):
    if TEST_MORE_METHODS > 1:
        SELECTION_METHOD = i % TEST_MORE_METHODS

    if TEST_TWO_SAVE_STRONGEST == True:
        if i == 2*TEST_MORE_METHODS-1:
            SAVE_STRONGEST = 0

    logInfo()
    output = np.zeros((ITERATIONS, dimension), dtype=int,)

    for j in range(0, ITERATIONS):
        print("Evaluate: {}".format(j))
        log.info(("#Evaluate: {}".format(j)))
        wynik = geneticAlgorithm()
        output[j] = wynik
        #print("Min cost of iteration:", costFunction(output[j], COMBINATIONS))
        log.info(getDate())
        log.info(("Min cost of iteration: {}".format(costFunction(output[j], COMBINATIONS))))
        log.info(("Min of all evaluations: {}".format(min(getCostsVector(output[0:j+1,:])))))
        #print("Min of all evaluations:", min(getCostsVector(output[0:j+1,:])))

    costVector = getCostsVector(output)
    plt.show()
    log.info("cost Vector")
    log.info(costVector)
    log.info("Minimum cost")
    log.info(min(costVector))
    log.info("of Individual:")
    log.info(output[costVector.tolist().index(min(costVector))])
    log.info("----------------------------")
    results.append(min(costVector))
log.info("Cost Vector of outputs:")
log.info(''.join(str(e)+', ' for e in results))
log.info("BEST OF ALL")
log.info(min(results))
log.info("OF ID:")
log.info((results.index(min(results))))



