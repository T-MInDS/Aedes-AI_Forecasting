import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def format_finetuning_samples(data, scaler):
    cols = ['Avg_Temp', 'Precip', 'Humidity', 'Ref']
    X = []
    y = []
    loc_times = []
    locs = data.Location.unique()
    data['Datetime'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
    for loc in locs:
        subset = data[data.Location.str.contains(loc)]
        for i in range(0, len(subset)-90):
            to_append = True
            if loc == 'Ceiba':
                missing_start = datetime(2017, 11, 12)
                missing_end = datetime(2018, 4, 17)
                if (max(missing_start, subset.Datetime.iloc[i]) <= min(missing_end, subset.Datetime.iloc[i]+timedelta(days=90))):
                    to_append = False
            if to_append:
                sample = subset.iloc[i:i+90]
                scaled = scaler.transform(sample[cols].values)

                X.append(scaled[:, 0:-1])
                y.append([scaled[-1, -1]])
                loc_times.append(sample.iloc[-1, 0:4])

    return np.array(X), np.array(y), np.array(loc_times)
