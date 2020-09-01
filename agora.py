import numpy as np
from agent import agent
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from agoraPlot import agoraPlot

#This script contains the class defining the market environment, which itself invokes the agent class in order to simulate
#the interaction of a number of agents in the physical market.

# objects of market:
# nbrs: NearestNeighbors object
# positions: list of all agent positions
# groups: list of all agent groups
# agents: List of agent objects
# week: Number of days in a consumption week
# Ng: Existing production groups
# needs: how much of each production group does an ave.consumer need per week
# prodCosts: production costs per production group, evolves with time

class agora:
    """Class defining the market environments in which agents interact. Inputs:
    population: Number of agents in market
    size: Physical size of market place
    Ng: Number of production groups in market
    prices: List setting the initial price for each group product
    eta_p: Random fluctuation to be added to each price, depending on the agent
    prodCosts: Base value of production costs of each product
    t_prod: Temporal variability of each production cost
    quantity: Initial stock of each product
    cash: Initial cash per agent
    eta_c: Random fluctuation to be added to each initial cash amount, depending on the agent
    eta_prod: Random fluctuoation to be added to each production cost, depending on the agent
    week: Number of days in a consumption week
    needs: List of how much of each production group does an ave consumer need per week
    rBuy: Base value for search radius for consumer inquiry
    eta_buy: Random fluctuation for rBuy, depending on the agent
    rSell: Base value for search radius for seller inquiry
    eta_sell: Random fluctuation for rSell, depending on the agent
    """
    def __init__(self, population=500, size=10, Ng=5, prices=[10, 10, 10, 10, 10], eta_p=[0, 0, 0, 0, 0], prodCosts=[5, 5, 5, 5, 5],
                 t_prod=[5, 5, 5, 5, 5], quantity=[50, 50, 50, 50, 50], cash=2000, eta_c=0, eta_prod=0.5, week=10,
                 needs=[10, 10, 10, 10, 10], rBuy=1.5, eta_buy=0, rSell=1.5, eta_sell=0):
        self.population = population
        self.size = size
        self.Ng = Ng
        self.prices = prices
        self.eta_p = eta_p
        self.baseCosts = prodCosts
        self.variabCosts = t_prod
        self.quantity = quantity
        self.cash = cash
        self.eta_c = eta_c
        self.eta_prod = eta_prod
        self.week = week
        self.needs = needs
        self.rBuy = rBuy
        self.eta_buy = eta_buy
        self.rSell = rSell
        self.eta_sell = eta_sell

        self.positions = np.array([0, 0])
        self.groups = []
        self.agents = []
        #Initialize population of buyers/sellers
        for i in range(population):
            position = np.random.uniform(low=0, high=size, size=(1, 2))
            self.positions = np.vstack((self.positions, position))

            group = np.random.choice(np.arange(Ng))
            self.groups.append(group)

            cash = self.cash + np.random.uniform(-1, 1) * self.eta_c
            price = self.prices[group] + np.random.uniform(-1, 1) * self.eta_p[group]
            quantity = self.quantity[group]
            prod = np.random.uniform(-1, 1) * self.eta_prod
            rBuy = self.rBuy + np.random.uniform(-1, 1) * self.eta_buy
            rSell = self.rSell + np.random.uniform(-1, 1) * self.eta_sell

            ag = agent(self, cash, price, quantity, position, group=group, prod=prod, rSell=rSell, rBuy=rBuy)
            self.agents.append(ag)

        self.positions = self.positions[1:, :]
        self.nbrs = NearestNeighbors(radius=max(self.rBuy + eta_buy, self.rSell + eta_sell)).fit(self.positions)

        self.day = 0
        self.T = self.week * 3
        self.prodCosts = [i + j * np.sin(2 * np.pi * self.day / self.T + 2 * np.pi * k / self.Ng) for i, j, k in zip(self.baseCosts, self.variabCosts, range(self.Ng))]

        #Contacts shows each transaction taking place. 1st column shows time instant, Second shows buyer, Third shows seller
        self.contacts = np.array([0, 0, 0])

    def marketDay(self):
        self.day += 1

        self.prodCosts = [i + j * np.sin(2 * np.pi * self.day / self.T + 2 * np.pi * k / self.Ng) for i, j, k in zip(self.baseCosts, self.variabCosts, range(self.Ng))]
        # self.prodCosts = [i + np.random.uniform(-1, 1) * j for i, j in zip(self.baseCosts, self.variabCosts)]

        order = np.arange(self.population)
        np.random.shuffle(order)
        #Open all stores in random order
        for i in order:
            ag = self.agents[i]
            ag.openStore()

        order = np.arange(self.population)
        np.random.shuffle(order)
        # Let agents buy in random order
        for i in order:
            ag = self.agents[i]
            reset = self.day % self.week == 0
            ag.shoppingRoutine(reset)

    def run(self, t=70, track=1):
        self.pricesT = np.zeros((self.population, t))
        self.cashT = np.zeros((self.population, t))
        self.stockT = np.zeros((self.population, t))
        self.aveNeeds = np.zeros((t, self.Ng))
        self.avePrices = np.zeros((t, self.Ng))
        self.aveCash = np.zeros((t, self.Ng))
        self.aveStock = np.zeros((t, self.Ng))
        self.costsT = np.zeros((t, self.Ng))

        self.tracker = np.zeros((t, self.Ng + 3)) #Record of each agent with its price, cash, stock, needs

        for i in range(t):
            aveNeeds = np.array([0 for _ in self.needs])
            for j, ag in enumerate(self.agents):
                self.pricesT[j, i] = ag.price
                self.cashT[j, i] = ag.cash
                self.stockT[j, i] = ag.stock
                aveNeeds += np.array(ag.consumerHierarchy)

                if j == track:
                    data = np.hstack((np.array([ag.price, ag.cash, ag.stock] + ag.consumerHierarchy)))
                    self.tracker[i, :] = data

            aveNeeds = aveNeeds / self.population
            self.aveNeeds[i, :] = aveNeeds

            for j in range(self.Ng):
                self.costsT[i, j] = self.prodCosts[j]
                same = [g==j for g in self.groups]
                prices = self.pricesT[same, i]
                self.avePrices[i, j] = np.mean(prices)
                cash = self.cashT[same, i]
                self.aveCash[i, j] = np.mean(cash)
                stock = self.stockT[same, i]
                self.aveStock[i, j] = np.mean(stock)

            self.marketDay()
        self.contacts = self.contacts[1:, :]



market = agora()
market.run()
plot = agoraPlot(market)
plot.plotTracker()
plot.plotPrices()
plot.plotStock()
plot.plotCash()
plot.plotPerGroup()
for t in range(3):
    plot.snapPlot(t)
plt.show()