import numpy as np
import torch
import torch.nn as nn
import random

#This script contains the definition of the agent class, as well as all the functions concerning agent actions.

class DQNetwork(nn.Module):
    """This class is used to form neural networks of size expressed in list Layers"""
    def __init__(self, Layers):
        super().__init__()
        self.f = nn.ModuleList()

        for i, j in zip(Layers, Layers[1:]):
            linear = nn.Linear(i, j)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.f.append(linear)

        self.relu = nn.ReLU()

    def forward(self, x):
        for i, f in enumerate(self.f):
            if i < len(self.f) - 1:
                x = self.relu(f(x))
            else:
                x = f(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity=100000, batch_size=100, weight=False, reward_position=3):
        self.capacity = capacity
        self.memory = []
        self.batch_size = batch_size
        self.rewards = []
        self.weight = weight
        self.reward_position = reward_position

    def push(self, replay):
        self.memory.append(replay)
        rew = replay[-self.reward_position].item()
        self.rewards.append(rew)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            del self.rewards[0]

    def sample(self):
        if self.weight:
            p = np.array(self.rewards.copy())
            if np.min(p) < 0:
                if len(p) > 1:
                    p = p - np.min(p)
                else:
                    p = abs(p)
            p = p / sum(p)
            take = np.random.choice(np.arange(len(self.memory)), size=min(len(self.memory), self.batch_size), p=p)
        else:
            take = np.random.choice(np.arange(len(self.memory)), size=min(len(self.memory), self.batch_size))
        give = [self.memory[i] for i in take]
        return give

    def __len__(self):
        return len(self.memory)

class nnInterface:
    """This class serves as a buffer between all replay buffers and all neural networks and the functions in class agent"""
    def __init__(self, LayersPrice=[3, 25, 25, 7], LayersStock=[3, 25, 25, 5], gamma=0.9, lr=0.1, based=False, nets=[None, None]):
        self.LayersPrice = LayersPrice
        self.LayersStock = LayersStock
        self.gamma = gamma
        self.lr = lr

        if based:
            self.nnPrice = nets[0]
            self.nnStock = nets[1]
        else:
            self.nnPrice = DQNetwork(LayersPrice)
            self.nnStock = DQNetwork(LayersStock)

        self.bufferPrice = ReplayBuffer()
        self.optPrice = torch.optim.Adam(self.nnPrice.parameters(), lr=lr)
        self.outputsPrice = np.linspace(1, 2, LayersPrice[-1])

        self.bufferStock = ReplayBuffer()
        self.optStock = torch.optim.Adam(self.nnStock.parameters(), lr=lr)
        self.outputsStock = np.linspace(0, 1, LayersStock[-1])

        self.criterion = nn.MSELoss()

    def defineStatePrice(self, agent):
        """Based on an agent object, this function defines its state to input its Q NN"""
        self.price = agent.price0
        st1 = (np.mean(agent.competitorPrices) - agent.price0) / (agent.price0 + 0.00001)
        st2 = (agent.costPerUnit - agent.cost0) / (agent.cost0 + 0.00001)
        st3 = (agent.demand - agent.demand0 + 0.00001) / (agent.demand0 + 0.00001)
        self.statePrice = np.array([st1, st2, st3])

    def defineStateStock(self, agent):
        """Based on an agent object, this function defines its state to input its Q NN"""
        st1 = (agent.cash - agent.cash0) / (agent.cash0 + 0.00001)
        st2 = (agent.stock - agent.demand + 0.00001) / (agent.demand + 0.00001)
        self.stateStock = np.array([st1, st2])

        profit = agent.cash - agent.cash0
        available = profit * (1 - agent.save)
        self.maxProduction = np.floor(available / agent.costPerUnit)

    def computePrice(self, epsilon=0.5):
        """Taking a q vector from the NN, this function outputs its action corresponding to an exploit or explore scheme"""
        self.statePrice0 = self.statePrice
        q = self.nnPrice(torch.tensor(self.statePrice).float())
        if random.random() < epsilon:
            action = torch.argmax(q)
        else:
            action = torch.randint(0, len(self.outputsPrice), (1,))[0]

        self.actionPrice = action
        self.price = self.outputsPrice[action.item()] * self.price
        return self.price

    def computeStock(self, epsilon=0.5):
        """Taking a q vector from the NN, this function outputs its action corresponding to an exploit or explore scheme"""
        self.stateStock0 = self.stateStock
        q = self.nnStock(torch.tensor(self.stateStock).float())
        if random.random() < epsilon:
            action = torch.argmax(q)
        else:
            action = torch.randint(0, len(self.outputsStock), (1,))[0]

        self.actionStock = action
        self.canMake = np.floor(self.outputsStock[action.item()] * self.maxProduction)
        return self.canMake

    def updateQ(self, agent, buffer, state0, state, action, net, optimizer):
        """This function takes into account the result of a recently set price in order to update the NN"""
        #Reward is the profit per produced good of the taken move
        profit = agent.cash - agent.cash0
        produced = agent.canMake
        rew = profit / (produced + 0.00001)

        buffer.push((state0, action.unsqueeze(0), torch.tensor(rew).unsqueeze(0), state))
        sample = buffer.sample()
        st, action, reward, st1 = zip(*sample)

        a = torch.cat(action).float().unsqueeze(1)
        r = torch.cat(reward).float().unsqueeze(1)
        qall = net(torch.tensor(st).float())
        q = torch.gather(qall, 1, a.long())
        qnext = net(torch.tensor(st1).float())
        qnext = qnext.detach().max(1)[0].unsqueeze(1)
        # qnext[done] = torch.tensor(0).float()

        loss = self.criterion(r + self.gamma * qnext, q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


class agentQ:
    """Class defining a consumer/producer whose producton is guided by Q Learnubg. Inputs:
    market: market class containing all neighboring consumers
    cash: initial cash of agent
    price0: initial selling price of seller
    quantity0: initial stock
    position: position of agent within agora
    group: Producer clan
    prod: Deviation from base production cost
    rSell: Radius of proximity detecting demand and prices
    rBuy: Radius of proximity detecting supply and prices
    save: Ratio of cash to be saved each day
    epsilon: List of values for epsilon, [eps_Price, eps_Stock]
    based: Boolean indicating if agent decision making must be based on a new net
    nets: List of DQNetwork objects, [nnPrice, nnStock]
    """
    def __init__(self, market, cash0, price0, quantity0, position, group=0, prod=0.01, rSell=1.1, rBuy=1.1, save=0.25, epsilon=[0.5, 0.5], based=False, nets=[None, None]):
        self.market = market
        self.cash = cash0
        self.price = price0
        self.stock = quantity0
        self.position = position
        self.group = group
        self.prod = prod + 1
        self.rSell = rSell
        self.rBuy = rBuy
        self.save = save
        self.epsilon = epsilon

        self.nnInterface = nnInterface(based=based, nets=nets)

        self.demand0 = self.market.needs[group]
        self.cash0 = cash0
        self.cost0 = self.market.prodCosts[self.group] * self.prod
        self.price0 = price0

        # Array showing all transactions of agent.
        # First col: 1 if buy, -1 if sell, 0 is produce. Second col: Price, or production cost. Third col: Agent's Cash before transaction.
        # Fourth col: Sellers stock before transaction, or when producing, items produced. Fifth col: Product group
        self.transactions = np.array([[-1, 0, 0, 0, 0]])
        self.resetNeeds()

    def resetNeeds(self):
        """Reset consumers needs after a new week"""
        # consumerHierarchy is simply a list which, for each production group in the market, stores the number of items each consumer
        #requires per week
        self.consumerHierarchy = self.market.needs.copy()

    def openStore(self, basic=False, train=True):
        """Routine which each producer carries out every day"""
        self.seeDemandandCompetitors()
        self.setPrice(basic=basic, train=train)
        self.setQuantity(basic=basic, train=train)
        self.produce()
        self.coherenceCheck()

    def seeDemandandCompetitors(self):
        """Examine neighboring consumers in order to see what the neighboring demand score is. Also examine neighboring
        competitor producers to see what the selling price usually is."""
        _, nbrs = self.market.nbrs.radius_neighbors(X=self.position, radius=self.rSell, sort_results=True)
        nbrs = nbrs[0][1:]
        self.demand = 0
        for n in nbrs:
            self.demand += self.market.agents[n].consumerHierarchy[self.group]

        competitors = [n for n in nbrs if self.market.groups[n] == self.group]
        self.competitorPrices = [self.market.agents[p].price for p in competitors]
        self.costPerUnit = self.market.prodCosts[self.group] * self.prod

    def setPrice(self, maxPercentile=50, basic=False, train=True):
        """Here is were we definitely model reality. Set a price based on how much demand has varied wrt the previous day,
        and a maxPercentile of the price within the prices of neighboring competitors."""

        self.nnInterface.defineStatePrice(self)
        if train:
            self.nnInterface.updateQ(self, self.nnInterface.bufferPrice, self.nnInterface.statePrice0,
                                     self.nnInterface.statePrice, self.nnInterface.actionPrice,
                                     self.nnInterface.nnPrice, self.nnInterface.optPrice)

        if basic:
            demandChange = 1 + (self.demand - self.demand0 + 0.00001) / (self.demand0 + 0.00001)
            basePrice = self.costPerUnit * demandChange

            if len(self.competitorPrices) > 0:
                competitivePrice = np.percentile(self.competitorPrices, maxPercentile)
                desired = max(basePrice, competitivePrice)
            else:
                desired = basePrice

            self.price = max(self.costPerUnit, desired)

            growth = self.price / self.price0
            self.nnInterface.actionPrice = torch.tensor(np.abs(self.nnInterface.outputsPrice - growth).argmin())
        else:
            self.price = self.nnInterface.computePrice(self.epsilon[0])

        self.demand0 = self.demand
        self.cost0 = self.costPerUnit
        self.price0 = self.price

    def setQuantity(self, basic=False, train=True):
        """Simply determine the quantity to produce based on current stock, existing demand, and available cash"""

        self.nnInterface.defineStateStock(self)
        if train:
            self.nnInterface.updateQ(self, self.nnInterface.bufferStock, self.nnInterface.statePrice0,
                                     self.nnInterface.statePrice, self.nnInterface.actionPrice,
                                     self.nnInterface.nnPrice, self.nnInterface.optPrice)

        if basic:
            profit = self.cash - self.cash0
            available = profit * (1 - self.save)
            toMake = self.demand - self.stock

            if toMake < 0:
                self.canMake = 0
            else:
                self.canMake = min(toMake, np.floor(available / self.costPerUnit))

            growth = self.canMake / np.floor(available / self.costPerUnit)
            self.nnInterface.actionStock = torch.tensor(np.abs(self.nnInterface.outputsStock - growth).argmin())
        else:
            self.canMake = self.nnInterface.computeStock(self.epsilon[1])

        self.cash0 = self.cash

    def produce(self):
        """Update existing cash and stock based on production"""
        if self.canMake > 0:
            self.cash -= self.canMake * self.costPerUnit
            self.stock += self.canMake
            self.transactions = np.vstack((self.transactions, np.array([0, self.costPerUnit, self.cash + self.canMake * self.costPerUnit, self.canMake, self.group])))

    def sell(self):
        """Update existing cash and stock based on selling"""
        self.cash += self.price
        self.stock -= 1

        self.transactions = np.vstack((self.transactions, np.array([-1, self.price, self.cash - self.price, self.stock + 1, self.group])))

    def shoppingRoutine(self, reset=False):
        """Shopping routine of consumer"""
        self.seeSupply()
        self.buyAsNeeded()
        self.coherenceCheck()

        if reset:
            self.resetNeeds()

    def seeSupply(self):
        """For each need of the consumer, see neighboring producer with the cheapest price"""
        _, nbrs = self.market.nbrs.radius_neighbors(X=self.position, radius=self.rBuy, sort_results=True)
        nbrs = nbrs[0]
        self.myself = nbrs[0]
        nbrs = nbrs[1:]
        self.cheapest = []
        for g in range(self.market.Ng):
            if g == self.group: #The fisherman can get fish himself, at production cost
                if self.stock > 0:
                    self.cheapest.append(self.myself)
                else:
                    self.cheapest.append(-1)
            else:
                producers = [n for n in nbrs if self.market.groups[n] == g]
                if len(producers) > 0:
                    prices = [self.market.agents[p].price for p in producers]
                    stock = [self.market.agents[p].stock for p in producers]
                    new = -1
                    for p in np.argsort(prices):
                        if stock[p] > 0:
                            new = producers[p]
                            break
                    self.cheapest.append(new)
                else:
                    self.cheapest.append(-1)

    def buyAsNeeded(self):
        good = -1
        sorted = np.argsort(self.consumerHierarchy)[::-1]
        for i in sorted:
            if self.cheapest[i] != -1 and self.consumerHierarchy[i] > 0:
                seller = self.cheapest[i]
                price = self.market.agents[seller].price
                if self.cash > price:
                    good = i
                    break

        if good != -1:
            self.market.agents[seller].sell()
            self.cash -= price
            self.consumerHierarchy[good] -= 1
            self.transactions = np.vstack((self.transactions, np.array([1, price, self.cash - price, self.market.agents[seller].stock + 1, good])))
            self.market.contacts = np.vstack((self.market.contacts, np.array([self.market.day, self.myself, seller])))

    def coherenceCheck(self):
        if self.cash < 0:
            print('Agent cash is negative.')
            print(self.transactions[-5:, :])

        if self.stock < 0:
            print('Agent stock is negative.')
            print(self.transactions[-5:, :])
