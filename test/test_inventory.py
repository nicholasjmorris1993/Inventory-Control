import pandas as pd
import sys
sys.path.append("/home/nick/Inventory-Control/src")
from inventory import qr


data = pd.read_csv("/home/nick/Inventory-Control/test/inventory.csv")
products = pd.unique(data["Product"])
models = dict()

for p in products:
    df = data.copy().loc[data["Product"] == p].drop(columns="Product").reset_index(drop=True)

    print(f"\n---- {p} ----\n")
    models[p] = qr(df, name=p)
    print("\nInventory Policy:")
    print(models[p].data)
