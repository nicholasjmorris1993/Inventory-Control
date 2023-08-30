import pandas as pd
import sys
sys.path.append("/home/nick/QR-Model/src")
from inventory import qr


data = pd.read_csv("/home/nick/QR-Model/test/inventory.csv")

model = qr(data)
print("\n")
print(model.data)
