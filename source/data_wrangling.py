import numpy as np

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

def pool_data_variables(ds, variables):
    """
    Function that takes a trajectory based xarray dataset and returns an array
    (without time or lat/lon indexing)  of all of the values in the dataset. 
    "pooling" all of the values into one data array for each of the specified variables, making sure this is
    done in the correct order and saving the result as a n-d matrix to keep coherence between variable and flags 

    Parameters
    ----------

        ds : xr.Dataset 
            array containing the data for the variable you want to pool ds[var name]
        
        variables : list(str)
            list contaning the variables 
    Returns:
    --------

        pooled data : np.array  
            array containing all data values for that variable for all trajectories
            shape number of data points x number variables
    """
    data_ds_temp = []
    num_vars = len(variables)
    num_points = len(ds.time_temp.values)

    for traj in range(ds["trajectory"].size):
        temp = ds.isel(trajectory=traj)
        temp = temp[~np.isnan(temp)]
        for val in temp.values:
            data_ds_temp.append(val)

    return np.array(data_ds_temp)