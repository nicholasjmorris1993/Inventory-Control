import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


FIG_SIZE = (15, 15)  # size of the plots
FONT_SIZE = 20  # font size of text in plots

def qr(df, name="QR Model"):
    model = QR()
    model.converge(df, name)
    model.simulate()
    model.plots()

    return model


class QR():
    def converge(self, df, name):
        print("\nQR Policy:\n")

        self.data = df.copy()
        self.name = name

        # collect the data
        self.demand = self.data.loc[self.data["Metric"] == "Average Demand", "Value"].values[0]  # weekly
        self.deviation = self.data.loc[self.data["Metric"] == "Standard Deviation Of Demand", "Value"].values[0]  # weekly
        self.lead_time = self.data.loc[self.data["Metric"] == "Lead Time", "Value"].values[0]  # weekly
        self.unit_cost = self.data.loc[self.data["Metric"] == "Order Unit Cost", "Value"].values[0]  # dollars
        self.stockout_cost = self.data.loc[self.data["Metric"] == "Stockout Unit Cost", "Value"].values[0]  # dollars
        self.order_cost = self.data.loc[self.data["Metric"] == "Order Cost", "Value"].values[0]  # dollars
        self.holding_cost = self.data.loc[self.data["Metric"] == "Holding Unit Cost", "Value"].values[0]  # dollars

        # solve for Q and R until solutions converge
        i = 0
        iterate = True
        self.order_quantity = np.sqrt(2 * self.order_cost * self.demand * 52 / self.holding_cost)  # initial Q
        while iterate:
            i += 1

            # compute R
            prob_no_shortage = 1 - (self.order_quantity * self.holding_cost) / (self.stockout_cost * self.demand * 52)
            z_score = stats.norm.ppf(prob_no_shortage)
            self.reorder_point = (self.deviation * z_score + self.demand) * self.lead_time  # R

            # compute Q
            loss_function = stats.norm.pdf(z_score) - z_score * stats.norm.sf(z_score)
            expected_shortage = self.deviation * loss_function
            order_quantity = np.sqrt(2 * self.demand * 52 * (self.order_cost + self.stockout_cost * expected_shortage) / self.holding_cost)

            # check convergence
            if abs((self.order_quantity - order_quantity) / order_quantity) <= 0.01:
                iterate = False
            self.order_quantity = order_quantity

            print(f"Iteration: {i}, Order Quantity: {round(self.order_quantity, 0)}, Reorder Point: {round(self.reorder_point, 0)}")

        self.order_frequency = self.order_quantity / self.demand
        self.order_quantity = round(self.order_quantity, 0)
        self.reorder_point = round(self.reorder_point, 0)

        # add solution to the data
        df = pd.DataFrame({
            "Metric": ["Order Quantity", "Reorder Point", "Order Frequency"],
            "Value": [self.order_quantity, self.reorder_point, self.order_frequency],
            "Unit": ["Products", "Products", "Weekly"],
        })
        self.data = pd.concat([self.data, df], axis="index")
    
    def simulate(self):
        print("\nSimulating Inventory Policy Over The Next Two Years:\n")
        np.random.seed(0)

        # simulate demand for the next two years
        demand = np.abs(np.random.normal(
            self.demand, 
            self.deviation, 
            size=52 * 2,
        )).astype(int)

        # simulate inventory levels and costs for the next two years
        inventory = [self.order_quantity]  # start out with an order received
        total_cost = list()
        satisfaction = list()
        lead_time = 0
        order_placed = False
        for d in range(len(demand)):
            # order received
            if lead_time >= self.lead_time:
                inventory[d] = inventory[d] + self.order_quantity
                lead_time = 0
                order_placed = False

            # demand supplied
            inventory[d] = inventory[d] - demand[d]

            # stockout
            if inventory[d] < 0:
                stockout_cost = self.stockout_cost * abs(inventory[d])
                holding_cost = 0
                demand_satisfied = min(1, 1 - abs(inventory[d]) / demand[d])
                inventory[d] = 0
            else:
                stockout_cost = 0
                holding_cost = self.holding_cost * inventory[d]
                demand_satisfied = 1

            # reorder point
            if inventory[d] <= self.reorder_point:
                if not order_placed:
                    order_placed = True
                    order_cost = self.order_cost + self.unit_cost * self.order_quantity
                else:
                    order_cost = 0
            else:
                order_placed = False
                order_cost = 0
            
            if order_placed:
                lead_time += 1  # waiting for an order

            # update inventory levels and statistics
            inventory.append(inventory[d])
            satisfaction.append(demand_satisfied)
            total_cost.append(stockout_cost + holding_cost + order_cost)

        # save the results
        inventory.pop()
        self.simulation = pd.DataFrame({
            "Week": np.arange(52 * 2) + 1,
            "Inventory": inventory,
            "Demand": demand,
            "Demand Satisfaction": satisfaction,
            "Total Cost": total_cost,
        })
        self.simulation["Cumulative Total Cost"] = self.simulation["Total Cost"].cumsum()

        print(f"Cumulative Total Cost: {self.simulation['Total Cost'].sum()}")
        print(f"Average Demand Satisfaction: {self.simulation['Demand Satisfaction'].mean()}")

    def plots(self):
        print("\nPlotting Trends\n")

        # plot inventory levels over time
        self.line_plot(
            self.simulation, 
            x="Week", 
            y="Inventory", 
            title=f"{self.name}: Inventory Levels Over The Next Two Years", 
            save=True,
        )

        # plot demand satisfaction over time
        self.bar_plot(
            self.simulation, 
            x="Week", 
            y="Demand Satisfaction", 
            title=f"{self.name}: Demand Satisfaction Over The Next Two Years", 
            save=True,
        )

        # plot cumulative total cost over time
        self.line_plot(
            self.simulation, 
            x="Week", 
            y="Cumulative Total Cost", 
            title=f"{self.name}: Cumulative Total Cost Of The Next Two Years", 
            save=True,
        )

    def line_plot(self, df, x, y, title="Line Plot", save=False):
        xaxis = df[x].tolist()
        yaxis = df[y].tolist()

        fig = plt.figure(figsize = FIG_SIZE)
        plt.plot(xaxis, yaxis, color ="blue")
        
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)

        plt.rc("font", size=FONT_SIZE)

        if save:
            title = re.sub("[^A-Za-z0-9]+", "", title)
            plt.savefig(title + ".png")
        else:
            plt.show()

    def bar_plot(self, df, x, y, title="Bar Plot", save=False):
        xaxis = df[x].tolist()
        yaxis = df[y].tolist()

        fig = plt.figure(figsize = FIG_SIZE)
        plt.bar(xaxis, yaxis, color ="blue")
        
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)

        plt.rc("font", size=FONT_SIZE)

        if save:
            title = re.sub("[^A-Za-z0-9]+", "", title)
            plt.savefig(title + ".png")
        else:
            plt.show()
