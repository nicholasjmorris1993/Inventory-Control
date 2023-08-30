import numpy as np
import pandas as pd
from scipy import stats


def qr(df):
    model = QR()
    model.converge(df)

    return model


class QR():
    def converge(self, df):
        self.data = df.copy()

        # collect the data
        self.demand = self.data.loc[self.data["Metric"] == "Average Demand", "Value"].values[0]  # weekly
        self.deviation = self.data.loc[self.data["Metric"] == "Standard Deviation Of Demand", "Value"].values[0]  # weekly
        self.lead_time = self.data.loc[self.data["Metric"] == "Lead Time", "Value"].values[0]  # weekly
        self.unit_cost = self.data.loc[self.data["Metric"] == "Order Unit Cost", "Value"].values[0]  # dollars
        self.stockout_cost = self.data.loc[self.data["Metric"] == "Stockout Unit Cost", "Value"].values[0]  # dollars
        self.order_cost = self.data.loc[self.data["Metric"] == "Order Cost", "Value"].values[0]  # dollars
        self.holding_cost = self.data.loc[self.data["Metric"] == "Holding Unit Cost", "Value"].values[0]  # dollars

        # solve for Q and R until solutions converge
        print("\n")
        i = 0
        iterate = True
        self.order_quantity = np.sqrt(2 * self.order_cost * self.demand * 52 / self.holding_cost)  # initial Q
        while iterate:
            i += 1

            # compute R
            prob_no_shortage = 1 - (self.order_quantity * self.holding_cost) / (self.stockout_cost * self.demand * 52)
            z_score = stats.norm.ppf(prob_no_shortage)
            self.reorder_point = self.deviation * z_score + self.demand  # R

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
