# -*- coding: utf-8 -*-
import numpy as np
import sys, os
def control_contact(inputfile,outputfile):
    data = np.load(inputfile)
    print(data.shape)
    column_223 = data[:, 222]
    rows_to_delete = np.where(column_223 <= 1)
    filtered_data = np.delete(data, rows_to_delete, axis=0)
    print("after data shape£º", filtered_data.shape)
    np.save(outputfile, filtered_data)
control_contact(sys.argv[1],sys.argv[2])
