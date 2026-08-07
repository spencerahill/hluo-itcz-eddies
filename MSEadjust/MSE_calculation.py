import numpy as np 
import xarray as xr
import gc, logging, os
from netCDF4 import Dataset
import dask
from dask_jobqueue import PBSCluster
from dask.distributed import Client

#dask.config.set(temporary_directory='/glade/derecho/scratch/hcluo')
# Create a PBS cluster object
#cluster = PBSCluster(
#    job_name = 'dask-wk23-hpc',
#    cores = 128,
#    memory = '235GiB',
#    processes = 8,
#    local_directory = '/glade/derecho/scratch/hcluo/pbs.$PBS_JOBID',
#    resource_spec = 'select=1:ncpus=128:mpiprocs=8:mem=235GB',
#    queue = 'main',
#    walltime = '12:00:00',
#    account = 'UPRI0023',
#    interface = 'hsn0'
#)
#
##'lo', 'bond0', 'hsn0', 'enp65s0' derecho
##['lo', 'mgt', 'ext', 'eno2'] ncasper
#print(cluster.job_script())
#client = Client(cluster)
## Spin up workers on our PBS cluster
#cluster.scale(8)
#client.wait_for_workers(8)


def read_data(fnames, varname,temporal_resolution,spatial_resolution):
    ds = xr.open_mfdataset(fnames,concat_dim = 'time',combine='nested',decode_times=False)
    TIME = ds['time']
    builtin_temporal_resolution = (TIME[1]-TIME[0]).values
    builtin_spatial_resolution = (ds.longitude[1]-ds.longitude[0]).values
    if (len(ds.latitude) > 1) & (ds.latitude[0] > ds.latitude[1]):
        ds = ds.reindex(latitude=ds.latitude[::-1])
    ds = ds.sel(time = slice(TIME[0],TIME[-1],int(temporal_resolution/builtin_temporal_resolution)),
                #level = slice(100,1000),
                latitude = slice(-90,90,int(spatial_resolution/builtin_spatial_resolution)), 
                longitude = slice(0,360,int(spatial_resolution/builtin_spatial_resolution)))
    print(ds)
    
    data = ds[varname]
    print("Size of Variable = {:5.2f} GiB".format(ds.nbytes / 1024 ** 3))
    del ds 
    gc.collect()
    print("Finishing reading data")

    return data


if __name__ == "__main__":
    Cp = 1004.0
    Lv = 1.0*2.5e6
    g = 9.8

    fpath_in = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/"
    fpath_out = "/glade/derecho/scratch/hcluo/MSE/MSE_"
    varname1 = "Q"
    varname2 = "T"
    varname3 = "Z"
    years = np.arange(2023,2024)
    months = ["%02d" % x for x in np.arange(7,8)]
    days = ["%02d" % x for x in np.arange(1,32)]
    temporal_resolution = 6
    spatial_resolution = 0.5

    for i in years:
        for j in months:
            year_month = str(i)+j
            for k in days:
                fnames1 = "e5.oper.an.pl.128_133_q.ll025sc." \
                +year_month+k+"00_"+year_month+k+"23.nc"
            
                fnames2 = "e5.oper.an.pl.128_130_t.ll025sc." \
                +year_month+k+"00_"+year_month+k+"23.nc"
            
                fnames3 = "e5.oper.an.pl.128_129_z.ll025sc." \
                +year_month+k+"00_"+year_month+k+"23.nc"

                folder = os.listdir(fpath_in+year_month)
                if np.isin(fnames1,folder):
                    print(year_month+k)
                    q = read_data(fpath_in+year_month+"/"+fnames1,varname1,temporal_resolution,spatial_resolution)
                    t = read_data(fpath_in+year_month+"/"+fnames2,varname2,temporal_resolution,spatial_resolution)
                    z = read_data(fpath_in+year_month+"/"+fnames3,varname3,temporal_resolution,spatial_resolution)
                else:
                    continue
                
        
                MSE = Cp*t+Lv*q+z
                del t, q, z
                gc.collect()
            
                fout = fpath_out+year_month+k+"00_"+year_month+k+"18.nc"
                os.system("rm -r "+fout)
                ds_out = xr.Dataset(data_vars = {"MSE":(("time","level","latitude","longitude"),MSE.data)},
                                    coords = {"time":MSE.time,
                                              "level":MSE.level,
                                              "latitude":MSE.latitude,
                                              "longitude":MSE.longitude},
                                    attrs = {"description":str(temporal_resolution)+" hourly moist static energy",
                                             "unit":"J kg^-1"})
                ds_out.to_netcdf(path = fout) 
                del  MSE,ds_out




    
    