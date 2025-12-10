import numpy as np
import pandas as pd

def pool_data_array(da):
    """
    Function that takes a trajectory based xarray dataset and returns an array
    (without time or lat/lon indexing)  of all of the values in the dataset. 
    "pooling" all of the values into one data array.

    Parameters
    ----------

        da : xr.DataArray
            array containing the data for the variable you want to pool ds[var name]

    Returns:
    --------

        pooled data : np.array  
            array containing all data values for that variable for all trajectories
    """
    data_ds_temp = []

    for traj in range(da["trajectory"].size):
        temp = da.isel(trajectory=traj)
        temp = temp[~np.isnan(temp)]
        for val in temp.values:
            data_ds_temp.append(val)

    return np.array(data_ds_temp) 

def pool_data_variables(ds, variables,exclude,VERBOSE=False):
    """
    Function that takes a trajectory based xarray dataset and returns an array
    (without time or lat/lon indexing)  of all of the values in the dataset. 
    "pooling" all of the values into one data array for each of the specified variables, making sure this is
    done in the correct order and saving the result as a pandas dataframe to keep coherence between variable and flags 

    Parameters
    ----------

        ds : xr.Dataset 
            array containing the data for the variable you want to pool ds[var name]
        
        variables : list(str)
            list contaning the variables

        exclude : list(int)
            list containing bouy indexes to exclude 
    Returns:
    --------

        pooled data : pd.DataFrame
            containing the pooled dataset with columns named with the variables
            plus a column for the bouy index (not the bouy ID but the index used
            when calling .isel(trajectory=index) to access the bouy)
    """

    #count number of datapoints in the final dataset
    num_points = 0
    for traj in range(ds["trajectory"].size):
        if traj in exclude:
            continue
        temp = ds.isel(trajectory=traj).time_temp.values
        temp = temp[~np.isnat(temp)]
        num_points = num_points + len(temp)
        

    #create a new array to store the data
    num_vars = len(variables)
    dataset = np.full((num_points,num_vars+1),np.nan)
    row_start = 0
    total = 0

    #add variable data
    for traj in range(ds["trajectory"].size):
        if traj in exclude:
            continue
        
        if VERBOSE: print(f"Bouy nr. {traj}")

        ds_traj = ds.isel(trajectory=traj)
        time_traj = ds_traj["time_temp"].values

        #maintain labels for all datapoints so original dataset can be reconstructed
     
        ids = np.array([traj] * len(time_traj[~np.isnat(time_traj)]))
    
        if VERBOSE: print(len(time_traj[~np.isnat(time_traj)]))

        col = 0
        for variable in variables:  
            #since all these arrays are indexed in the same order I should be able to go
            #variable by variable and add column by column, but maybe check this 
            
            data_var_val = ds_traj[variable].values
            data_var_val = data_var_val[~np.isnat(time_traj)]
        
            if col == 0: total = total + len(data_var_val)
            row_end = row_start + len(data_var_val)
            
            dataset[row_start:row_end,col] = data_var_val

            if col == 0 and VERBOSE == True: 
                print(f"number of points: {len(data_var_val)}")
                print(f"row start:{row_start}, row_end:{row_end}")

            col += 1

        if VERBOSE: print("====================================================")
        dataset[row_start:row_end,-1] = ids

        row_start = row_end
        
    dataset_pooled = pd.DataFrame(dataset,columns=variables+["KVS_ID"])

    return dataset_pooled