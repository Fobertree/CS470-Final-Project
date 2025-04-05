from polygon import RESTClient
from polygon.rest.models import Agg
import os
from dotenv import load_dotenv
import pandas as pd
import json

# https://polygon.io/docs/rest/crypto/aggregates/custom-bars
start_date = "2021-01-09"
end_date = "2023-02-10"

print(os.getcwd())
assert(os.path.exists("./Models/Data/btcusd.json"))

load_dotenv("./env")

POLY_API_KEY = os.getenv("POLY_API_KEY")

client = RESTClient(POLY_API_KEY)


aggs = []
for a in client.list_aggs(
    "X:BTCUSD",
    1,
    "day",
    "2024-04-06",
    "2025-04-03",
    adjusted="true",
    sort="asc",
    limit=120,
):
    aggs.append(a)


def aggs_to_csv(objs : list[Agg], csv_path :str | None = None):
    columns = ["open", "high", "low", "close", "volume", "vwap", "transactions", "otc"]
    ts_index = []
    data = []

    for agg in objs:
        ts_index.append(agg.timestamp)
        data.append([agg.open, agg.high, agg.low, agg.close, agg.volume, agg.vwap, agg.transactions, agg.otc])
    
    df = pd.DataFrame(data = data, columns = columns, index = ts_index)
    df.index.name = "timestamp"
    
    if csv_path:
        df.to_csv(csv_path)
    
    return df

aggs_to_csv(aggs, "./Models/Data/data.csv")