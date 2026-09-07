#calculated adjusted v
import numpy as np 
import xarray as xr
import gc, logging, os,copy,glob,sys
#import dask
#from dask_jobqueue import PBSCluster
#from dask.distributed import Client
import myfun as wf
import calendar
from scipy.integrate import quad
from windspharm.xarray import VectorWind

def read_data(fname,varname,temporal_resolution,spatial_resolution):
    ds = xr.open_mfdataset(fname,concat_dim = 'time',combine='nested',decode_times=False)
    TIME = ds['time']
    builtin_temporal_resolution = (TIME[1]-TIME[0]).values
    builtin_spatial_resolution = (ds.longitude[1]-ds.longitude[0]).values
    if (len(ds.latitude) > 1) & (ds.latitude[0] > ds.latitude[1]):
        ds = ds.reindex(latitude=ds.latitude[::-1])
    ds = ds.sel(time = slice(TIME[0],TIME[-1],int(temporal_resolution/builtin_temporal_resolution)),
                latitude = slice(-90,90,int(spatial_resolution/builtin_spatial_resolution)), 
                longitude = slice(0,360,int(spatial_resolution/builtin_spatial_resolution)))
    
    data = ds[varname]

    logging.info("finished reading data")
    return data.reindex(latitude=data.latitude[::-1])

def read_mean_flux(fpath,fname,varname, year, month,temporal_resolution,spatial_resolution):
    fnames = []
    MONTHS = ["%02d" % x for x in np.arange(1,13)]

    if MONTHS[month-1] == '01':
        fnames.append(fpath+str(year-1)+"12/"+fname\
        +str(year-1)+"121606_"+str(year)+"010106.nc")
    else:
        fnames.append(fpath+str(year)+MONTHS[month-2]+"/"+fname\
        +str(year)+MONTHS[month-2]+"1606_"+str(year)+MONTHS[month-1]+"0106.nc")

    fname1 = fpath+str(year)+MONTHS[month-1]+"/"+fname\
    +str(year)+MONTHS[month-1]+"0106_"+str(year)+MONTHS[month-1]+"1606.nc"
    
    fnames.append(fname1)
    
    if MONTHS[month-1] == '12':
        fname2 = fpath+str(year)+MONTHS[month-1]+"/"+fname\
        +str(year)+MONTHS[month-1]+"1606_"+str(year+1)+"010106.nc"
    else:
        fname2 = fpath+str(year)+MONTHS[month-1]+"/"+fname\
        +str(year)+MONTHS[month-1]+"1606_"+str(year)+MONTHS[month]+"0106.nc"

    fnames.append(fname2)

    if MONTHS[month-1] == '12':
        fnames.append(fpath+str(year+1)+"01/"+fname\
        +str(year+1)+"010106_"+str(year+1)+"011606.nc")
    
    ds = xr.open_mfdataset(fnames,concat_dim = 'forecast_initial_time', combine='nested',decode_times=False)
    
    #slicing and regridding
    if (len(ds.latitude) > 1) & (ds.latitude[0] > ds.latitude[1]):
        ds = ds.reindex(latitude=ds.latitude[::-1])

    builtin_resolution = (ds.longitude[1]-ds.longitude[0]).values
    ds = ds.sel(forecast_hour = slice(temporal_resolution,12,temporal_resolution),
                latitude = slice(-90,90,int(spatial_resolution/builtin_resolution)), 
                longitude = slice(0,360,int(spatial_resolution/builtin_resolution)))
    
    de = ds.stack(time=("forecast_initial_time","forecast_hour"))
    data = de[varname]
    data = data.transpose("time", "latitude", "longitude")
    
    Data = xr.DataArray(data.values, dims=['time','latitude','longitude'], 
                            coords=dict(
                            latitude=data.latitude,
                            longitude=data.longitude,
                            time=ds.forecast_initial_time.values[0]+temporal_resolution*np.arange(1,1+len(data.time))))
    
    logging.info("finished reading data")
    return Data


def column_integrate(data, mask):
    return wf.nantrapz(xr.where(mask==1, data,np.nan),data.level,dim = "level")/9.8*100.0


def maskout(model,sp):
    if "level" in model.coords:
        level_exp,sp_exp = xr.broadcast(model.level, sp)
        level_exp = level_exp.transpose(*model.dims)
        sp_exp = sp_exp.transpose(*model.dims)
        Data = xr.DataArray(np.ones(np.shape(model)),dims = model.dims,coords = model.coords)
        mask = xr.where((level_exp <= sp_exp), Data,0)
        logging.info("finished masking")
        return mask

    else:
        raise ValueError("stopped masking, the level coord is not here")

def divergence(u,v):
    div_H = VectorWind(u, v).divergence()
    logging.info("finished calculating divergence")
    return div_H

def flux(model,year,month):
    fpath_in  = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.fc.sfc.meanflux/"
    varname1 = "MSSHF"
    varname2 = "MSLHF"
    varname3 = "MSNSWRF"
    varname4 = "MSNLWRF"
    varname5 = "MTNSWRF"
    varname6 = "MTNLWRF"

    fname1 = "e5.oper.fc.sfc.meanflux.235_033_msshf.ll025sc."
    fname2 = "e5.oper.fc.sfc.meanflux.235_034_mslhf.ll025sc."
    fname3 = "e5.oper.fc.sfc.meanflux.235_037_msnswrf.ll025sc."
    fname4 = "e5.oper.fc.sfc.meanflux.235_038_msnlwrf.ll025sc."
    fname5 = "e5.oper.fc.sfc.meanflux.235_039_mtnswrf.ll025sc."
    fname6 = "e5.oper.fc.sfc.meanflux.235_040_mtnlwrf.ll025sc."

    spatial_resolution  = abs(model.longitude[0] - model.longitude[1])
    temporal_resolution = abs(model.time[0] - model.time[1])
    temporal_resolution = int(temporal_resolution/2)

    sensible = read_mean_flux(fpath_in,fname1,varname1, year, month, temporal_resolution, spatial_resolution).sel(time = model.time)
    latent   = read_mean_flux(fpath_in,fname2,varname2, year, month, temporal_resolution, spatial_resolution).sel(time = model.time)
    SS       = read_mean_flux(fpath_in,fname3,varname3, year, month, temporal_resolution, spatial_resolution).sel(time = model.time)
    SL       = read_mean_flux(fpath_in,fname4,varname4, year, month, temporal_resolution, spatial_resolution).sel(time = model.time)
    TS       = read_mean_flux(fpath_in,fname5,varname5, year, month, temporal_resolution, spatial_resolution).sel(time = model.time)
    TL       = read_mean_flux(fpath_in,fname6,varname6, year, month, temporal_resolution, spatial_resolution).sel(time = model.time)
    F = TS+TL-SS-SL-latent-sensible 

    logging.info("finished calculating NEI")
    return F

def adjust(mask,divergence):
    if divergence.latitude[0] < divergence.latitude[1]:
        divergence = divergence.reindex(latitude=divergence.latitude[::-1])

    wind = VectorWind(divergence, divergence)
    div_adj_spec = wind._api.s.grdtospec(divergence.transpose("latitude", "longitude", "time"))

    vort_adj_spec = np.zeros_like(div_adj_spec)
    u_adj, v_adj = wind._api.s.getuv(vort_adj_spec, div_adj_spec)
                    
    #u_adj = (xr.ones_like(divergence.transpose("latitude", "longitude", "time")) * u_adj).transpose(*divergence.dims)
    #v_adj = (xr.ones_like(divergence.transpose("latitude", "longitude", "time")) * v_adj).transpose(*divergence.dims)
    u_adj = xr.DataArray(u_adj,coords = divergence.transpose("latitude", "longitude", "time").coords).transpose(*divergence.dims)
    v_adj = xr.DataArray(v_adj,coords = divergence.transpose("latitude", "longitude", "time").coords).transpose(*divergence.dims)

    surface_pressure = (mask*(mask.level)).max(dim = "level")*100.0
    top_pressure = mask.level[0]*100.0
    u_adjust = u_adj/(surface_pressure-top_pressure)*9.8
    v_adjust = v_adj/(surface_pressure-top_pressure)*9.8

    logging.info("finished calculating adjustment")
    return u_adjust,v_adjust

def smooth(data):
    import metpy as mp
    out = mp.calc.smooth_n_point(data, n=9, passes=3)
    out = xr.DataArray(out, dims=['latitude','longitude'], 
                            coords=dict(
                            latitude=out.latitude,
                            longitude=out.longitude))
    return out

def write_data(fout,data_out,year,month):
    temporal_resolution = int(data_out[0].time[0]-data_out[0].time[1])
    ds_out = xr.Dataset(data_vars = {"V":(data_out[0].dims,data_out[0].data),
                                     "MSE":(data_out[1].dims,data_out[1].data)},
                        coords = data_out[0].coords,
                        attrs = {"description":str(temporal_resolution)+" hourly v & MSE calibrate with surface pressure",
                                 "period": str(year)+"%02d" % month,
                                 "unit":"m s^-1, J kg^-1"})                    
                    
    os.system("rm -r "+fout)
    ds_out.to_netcdf(path = fout) 
    logging.info("finished writing")

    return


def tendency(fnames,file_list,data,varname,mask):
    spatial_resolution  = abs(data.longitude[0] - data.longitude[1])
    temporal_resolution = abs(data.time[0] - data.time[1])
    time = data.time
    index_before = file_list.index(fnames[0])-1
    if index_before >= 0:
        data_before = read_data(file_list[index_before],varname,temporal_resolution,spatial_resolution)
        #fnames_extend = [file_list[index_before]] + fnames_extend
        data = xr.concat([data_before, data], dim='time')

    index_after  = file_list.index(fnames[-1])+1
    if index_after < len(file_list):
        data_after = read_data(file_list[index_after],varname,temporal_resolution,spatial_resolution)
        #fnames_extend.append(file_list[index_after])
        data = xr.concat([data,data_after], dim='time')

    tend = wf.ddt(column_integrate(data,mask)).sel(time = time)

    logging.info("finished calculating column tendency")
    return tend/3600.0



if __name__ == "__main__":
    begin = int(sys.argv[1])
    print(begin)
    logging.info(begin)
    years = range(begin,begin+2)
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    days = ["%02d" % x for x in np.arange(1,32)]
    temporal_resolution = 6
    spatial_resolution = 0.5
    fpath_in0 = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc/"
    fpath_in1 = "/glade/derecho/scratch/hcluo/MSE/"
    fpath_in2 = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/"
    fpath_out = "/glade/derecho/scratch/hcluo/v_adjust/adjustment_"
    file_pattern = fpath_in1+"MSE_*.nc"  # Adjust the path and file extension as needed
    file_list = sorted(glob.glob(file_pattern))

    varname0 = "SP"
    varname1 = "MSE"
    varname2 = "U"
    varname3 = "V"

    for i in years:
        for j in months:
            fnames0 = []
            fnames1 = []
            fnames2 = []
            fnames3 = []
            #fnames1_extend = []
    
            year_month = str(i)+"%02d" % j
            last_day = str(calendar.monthrange(i, j)[1])
    
            fname0 = "e5.oper.an.sfc.128_134_sp.ll025sc."\
                +year_month+"0100_"+year_month+last_day+"23.nc"
                
            fnames0.append(fpath_in0+year_month+"/"+fname0)
    
            for k in days:
                fname1 = "MSE_"+year_month+k+"00_"+year_month+k+"18.nc"
    
                fname2 = "e5.oper.an.pl.128_131_u.ll025uv." \
                +year_month+k+"00_"+year_month+k+"23.nc"
    
                fname3 = "e5.oper.an.pl.128_132_v.ll025uv." \
                +year_month+k+"00_"+year_month+k+"23.nc"
    
                folder = os.listdir(fpath_in1)
                if np.isin(fname1,folder):
                    fnames1.append(fpath_in1+fname1)
                    #fnames1_extend.append(fpath_in1+fname1)
                    fnames2.append(fpath_in2+year_month+"/"+fname2)
                    fnames3.append(fpath_in2+year_month+"/"+fname3)

            MSE = read_data(fnames1,varname1,temporal_resolution,spatial_resolution)
            u = read_data(fnames2,varname2,temporal_resolution,spatial_resolution)
            v = read_data(fnames3,varname3,temporal_resolution,spatial_resolution)      
            sp = read_data(fnames0,varname0,temporal_resolution,spatial_resolution).sel(time = MSE.time)/100.0            
            mask = maskout(MSE,sp)
            del sp
            gc.collect()
    
            uMSE_col = column_integrate(u*MSE,mask)
            vMSE_col = column_integrate(v*MSE,mask)
            div_H = divergence(uMSE_col,vMSE_col)

            F = flux(MSE,i,j)

            ddt = tendency(fnames1,file_list,MSE,varname1,mask)
        
            residue = ddt+div_H-F
        
            uMSE_adjust,vMSE_adjust = adjust(mask,residue)

            v_adj = v-vMSE_adjust/MSE 
            v_adj = xr.where((mask == 1), v_adj,np.nan)
            MSE = xr.where((mask == 1), MSE,np.nan)
            
            fout = fpath_out+year_month+".nc"
            write_data(fout,[v_adj,MSE],i,j)
            print(year_month)
    
    








