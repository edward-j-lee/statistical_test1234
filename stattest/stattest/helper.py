import pandas as pd
import numpy as np

from . import python_code

def handle_data_file(file):
    #data = file.read()
    #print (data)
    #with open("test.csv", "wb") as dest_file:
    #    dest_file.write(data)
    #    dest_file.close()
    #print ('sdfs',dest_file)
    data_frame =  pd.read_csv(file, header=None)
    series=np.asarray(data_frame).flatten()
    series=series[np.logical_not(np.isnan(series))]
    print(series)