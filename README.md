# QAgora
## Operating Principle
QAgora is a simulation of a physical free market, made up of agents which, being fixed in space, interact with their neighboring agents. On the one hand, each of these agents is a consumer. Given that there are a specified number of groups of production, each agent has to consume a certain number of goods from each production group during a whole week. To do so, each agent must buy goods from neighboring agents. On the other hand, each agent operates as a producer of a particular production group. As a producer, the agent's goal is to make as much profit as is possible, starting each market day by fixing a competitive selling price and producing the optimum amount of goods for such goal. The money which the agent makes by selling its own goods is the money which it spends satisfying its needs as a consumer. In this way, a simulation simply plays out a series of market days, each of these allowing each agent to sell and consume as needed.

As this is a physical market (like an old school greek Agora), agents can only interact with neighboring agents. This means that when buying, each agent will look for the cheapest option for its most needed good in its surroundings. When selling, each agent is also capable of looking at the selling price set by neighboring competitors as well as the demand of its own production good in its area. 

### How does an agent buy?
As has been said, every agent has to consume a specified number of goods from each production group per week. During each market day, the agent will buy from the good it needs the most from the neighboring agent selling that good at the cheapest price. In case it needs a good from its own production group, the agent will consume its own stock. For simplicity, each agent only has the possibility of buying a single good per market day.

### How does an agent sell?
How an agent sells is simply a product of what selling price it fixes and how much of its production good does it choose to produce at the start of each market day. To make this project particularly interesting, these two decisions are carried out by two neural networks which are trained during the simulation via Q-Learning.

The neural network in charge of setting the agent's selling price takes three variables as input: the normalized difference between the agent's previous market day's selling price and the neighboring competitors' selling prices, a normalized measure of how much the production cost of the agent's production group has varied in the past day, and a normalized measure of how much demand for the agent's production good in neighboring agents has varied in the past day. With these three variables, the neural network determines a new price by reducing or enlarging the selling price of the last market day. This neural network is trained by using the profit that is visible during the next market day as a reward.

The neural network in charge of determining how much of its own production good the agent produces is similar, but takes as an input a measure of how much the agent's available cash has varied with respect to the last market day, and how much the agent's current stock deviates from the existing neighboring demand for its production group. The output of the neural network is how much of its available cash should be invested in stocking up. For training, this neural network is also reinforced with the resulting profit.

## Script Description

**agentLearner.py:** This script contains the definition of the agent class, as well as all the functions concerning agent actions. Here is where every neural network is created and invoked for its training.

**agora.py:** This script contains the class defining the market environment, which itself invokes the agent class in order to simulate
the interaction of a number of agents in the physical market.

**agoraPlot.py:** This contains a number of functions in charge of the visualization of relevant simulation parameters.
