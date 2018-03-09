def rouletteMethod(population, costVector):
    #costVector = (costVector-min(costVector)+1)*1.5             #SHOULDNT BE DONE LIKE THIS
    sumCost = sum(costVector)
    adjVector = sumCost/costVector
    sumAdj = sum(adjVector)
    probabilityVector = adjVector/sumAdj
    print(sum(probabilityVector))
    parents = np.zeros(population.shape[0], dtype=int)
    newPopulation = np.zeros((population.shape[0], population.shape[1]), dtype=int)
    for i in range(0, population.shape[0]):
        parents[i] = pickOneRoulette(probabilityVector)
        newPopulation[i] = population[parents[i]]
    return newPopulation


def pickOneRoulette(probabilityVector):
    pick = random.uniform(0, 1)
    sum = 0
    for i in range(0, probabilityVector.shape[0]):
        sum += probabilityVector[i]
        if(sum >= pick):
            return i
        # else:
        #     sum += probabilityVector[i]
    return probabilityVector.shape[0]-1


def tournamentMethod(population, costVector):
    parents = np.zeros(population.shape[0], dtype=int)
    newPopulation = np.zeros((population.shape[0], population.shape[1]), dtype=int)

    for i in range(0, population.shape[0]):
        parents[i] = pickOneTournament(costVector)
        newPopulation[i] = population[parents[i]]

    return newPopulation


def pickOneTournament(costVector):
    tournamentTeam = np.zeros(TOUR, dtype=int)

    for i in range(0, tournamentTeam.shape[0]):
        tournamentTeam[i] = getRandomInvidivual()

    return costVector.tolist().index(min(costVector[tournamentTeam]))


def selection(population):
    costVector = getCostsVector(population)
    if SELECTION_METHOD == 0:
        next_population = rouletteMethod(population, costVector)
    if SELECTION_METHOD == 1:
        next_population = tournamentMethod(population, costVector)
    if(SAVE_STRONGEST == True):
        bestIndividual = population[costVector.tolist().index(min(costVector))]
        newCost = getCostsVector(next_population).tolist()
        worstIndividualIndex = newCost.index(min(newCost))
        next_population[worstIndividualIndex] = bestIndividual
    return next_population
