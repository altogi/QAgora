import numpy as np
import matplotlib.pyplot as plt
import imageio

# Here, plots regarding the simulation's evolution are presented.

class agoraPlot:
    def __init__(self, agora):
        self.market = agora

    def plotPerGroup(self):
        t = np.arange(self.market.pricesT.shape[1]) + 1
        for i in range(self.market.Ng):
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(16, 10)
            ax.set_xlabel('Market Days [-]', fontsize=24)
            ax.grid(True)
            ax.set_ylabel('Value Normalized w.r.t. Maximum [-]', fontsize=20)
            ax.tick_params(axis='both', labelsize=18)
            ax.set_title('Evolution of Production Group ' + str(i), fontsize=24)
            fig.tight_layout()

            ax.plot(t, self.market.aveCash[:, i] / np.max(self.market.aveCash[:, i]), label='Cash')
            ax.plot(t, self.market.avePrices[:, i] / np.max(self.market.avePrices[:, i]), label='Price')
            ax.plot(t, self.market.aveStock[:, i] / np.max(self.market.aveStock[:, i]), label='Stock')
            ax.plot(t, self.market.aveNeeds[:, i] / np.max(self.market.aveNeeds[:, i]), label='Demand')
            ax.plot(t, self.market.costsT[:, i] / np.max(self.market.costsT[:, i]), label='Production Cost')
            ax.legend(fontsize=20)

    def plotTracker(self):
        fig, ax = plt.subplots(4, 1)
        fig.set_size_inches(10, 25)
        fig.tight_layout()
        t = np.arange(self.market.tracker.shape[0]) + 1
        ylab = ['Seller Price [$]', 'Seller Cash [$]', 'Seller Stock [-]', 'Buyer Needs Per Group [-]']
        for i, a in enumerate(ax):
            a.set_xlabel('Market Days [-]', fontsize=15)
            a.grid(True)
            a.tick_params(axis='both', labelsize=13)
            a.set_ylabel(ylab[i], fontsize=15)
            a.set_title('Evolution of ' + ylab[i], fontsize=18)

            if i < 3:
                a.plot(t, self.market.tracker[:, i])
            else:
                for j in range(self.market.Ng):
                    a.plot(t, self.market.tracker[:, i + j], label='Production Group ' + str(j))
                a.legend(fontsize=13)

    def plotPrices(self):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(16, 10)
        ax.set_xlabel('Market Days [-]', fontsize=24)
        ax.grid(True)
        ax.set_ylabel('Price Per Production Group [$]', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title('Evolution of Price Per Production Group', fontsize=24)
        fig.tight_layout()
        t = np.arange(self.market.pricesT.shape[1]) + 1
        for i in range(self.market.Ng):
            ax.plot(t, self.market.avePrices[:, i], label='Production Group ' + str(i))
        ax.legend(fontsize=20)

    def plotCash(self):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(16, 10)
        ax.set_xlabel('Market Days [-]', fontsize=24)
        ax.grid(True)
        ax.set_ylabel('Cash Per Production Group [$]', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title('Evolution of Cash Per Production Group', fontsize=24)
        fig.tight_layout()
        t = np.arange(self.market.pricesT.shape[1]) + 1
        for i in range(self.market.Ng):
            ax.plot(t, self.market.aveCash[:, i], label='Production Group ' + str(i))
        ax.legend(fontsize=20)

    def plotStock(self):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(16, 10)
        ax.set_xlabel('Market Days [-]', fontsize=24)
        ax.grid(True)
        ax.set_ylabel('Stock Per Production Group [$]', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title('Evolution of Stock Per Production Group', fontsize=24)
        fig.tight_layout()
        t = np.arange(self.market.pricesT.shape[1]) + 1
        for i in range(self.market.Ng):
            ax.plot(t, self.market.aveStock[:, i], label='Production Group ' + str(i))
        ax.legend(fontsize=20)

    def plotNeeds(self):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(16, 10)
        ax.set_xlabel('Market Days [-]', fontsize=24)
        ax.grid(True)
        ax.set_ylabel('Average Need Per Production Group [-]', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title('Evolution of Average Need Per Production Group', fontsize=24)
        fig.tight_layout()
        t = np.arange(self.market.pricesT.shape[1]) + 1
        for i in range(self.market.Ng):
            ax.plot(t, self.market.aveNeeds[:, i], label='Production Group ' + str(i))
        ax.legend(fontsize=20)

    def plotCosts(self):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(16, 10)
        ax.set_xlabel('Market Days [-]', fontsize=24)
        ax.grid(True)
        ax.set_ylabel('Production Cost Per Group [$]', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title('Evolution of Production Cost Per Group', fontsize=24)
        fig.tight_layout()
        t = np.arange(self.market.pricesT.shape[1]) + 1
        for i in range(self.market.Ng):
            ax.plot(t, self.market.costsT[:, i], label='Production Group ' + str(i))
        ax.legend(fontsize=20)

    def snapPlot(self, type=0):
        """This method plots each two-dimensional domain separately, but joins all of them in a GIF animation which allows
        a 3D evolution of them to be visualized."""
        #0: Price, 1: Cash, 2:Stock
        if type == 0:
            data = self.market.pricesT
            title = 'Evolution of Local Market Prices'
        elif type == 1:
            data = self.market.cashT
            title = 'Evolution of Agent Cash'
        elif type == 2:
            data = self.market.stockT
            title = 'Evolution of Agent Stock'
        time = np.arange(data.shape[1])
        v0 = np.min(data)
        v1 = np.max(data)

        def update(t):
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 13)

            contacts = self.market.contacts[self.market.contacts[:, 0] == t + 1, 1:]
            for c in contacts:
                ax.plot(self.market.positions[c, 0], self.market.positions[c, 1], 'k--', 5)

            cm = plt.cm.get_cmap('viridis')
            sc = ax.scatter(self.market.positions[:, 0], self.market.positions[:, 1], c=data[:, t], cmap=cm, s=200,
                            vmin=v0, vmax=v1)

            ax.set_title('t = ' + str(t + 1), fontsize=24)
            ax.set_xlabel('x [m]', fontsize=18)
            ax.set_ylabel('y [m]', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=15)
            plt.axis('equal')
            cbar = plt.colorbar(sc)
            cbar.ax.tick_params(labelsize=15)
            ax.set_xlim(0, self.market.size)
            ax.set_ylim(0, self.market.size)

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

        kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
        imageio.mimsave(title + '.gif', [update(i) for i in time], fps=2)
