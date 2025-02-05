import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import datetime

ticker = input("ticker name")
years = input("years back: ")
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(years=years)
