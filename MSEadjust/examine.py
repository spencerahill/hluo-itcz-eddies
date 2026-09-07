#examine the adjustment
import numpy as np 
import xarray as xr
import gc, logging, os,copy,glob
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
    return Data.reindex(latitude=Data.latitude[::-1])


def column_integrate(data, mask):
    return wf.nantrapz(xr.where(mask==1, data,np.nan),data.level,dim = "level")/9.8*100.0


def maskout(model,sp):
    if "level" in model.coords:
        level_exp,sp_exp = xr.broadcast(model.level, sp)
        level_exp = level_exp.transpose(*model.dims)
        sp_exp = sp_exp.transpose(*model.dims)
        Data = xr.DataArray(np.ones(np.shape(sp_exp)),dims = sp_exp.dims,coords = sp_exp.coords)
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
    print(model.time)
    sensible = read_mean_flux(fpath_in,fname1,varname1, year, month, temporal_resolution, spatial_resolution)#.sel(time = model.time)
    print(sensible.time)
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
    ds_out = xr.Dataset(data_vars = {"uMSE":(data_out[0].dims,data_out[0].data),
                                     "vMSE":(data_out[1].dims,data_out[1].data)},
                        coords = data_out[0].coords,
                        attrs = {"description":str(temporal_resolution)+" hourly column-integrated adjusted horizontal MSE flux calibrate with surface pressure",
                                 "period": str(year)+"%02d" % month,
                                 "unit":"W m^-1"})                    
                    
    os.system("rm -r "+fout)
    ds_out.to_netcdf(path = fout) 
    logging.info("finished writing")

    return


def tendency(fnames,file_list,data,varname,mask):
    time = data.time
    spatial_resolution  = abs(data.longitude[0] - data.longitude[1])
    temporal_resolution = abs(time[0] - time[1])
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
    import sys
    np.set_printoptions(threshold=np.inf)
    mask = mask.sel(time = data.time)
    tend = wf.ddt(column_integrate(data,mask)).sel(time = time)

    logging.info("finished calculating column tendency")
    return tend/3600.0

def before(year,month):
    if month == 1:
        year = year-1
        month = 12
    else:
        month = month-1
    return str(year)+"%02d" % month

def after(year,month):
    if month == 12:
        year = year+1
        month = 1
    else:
        month = month+1
    return str(year)+"%02d" % month


def last_day_before(year,month):
    if month == 1:
        year = year-1
        month = 12
    else:
        month = month-1
    return str(calendar.monthrange(year, month)[1])

def last_day_after(year,month):
    if month == 12:
        year = year+1
        month = 1
    else:
        month = month+1
    return str(calendar.monthrange(year, month)[1])


def map_plot(Data,cmap,fileout):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
    import geocat.viz as gv

    data = Data[0]
    west_bd = min(data.longitude)
    east_bd = max(data.longitude)
    south_bd = min(data.latitude)
    north_bd = max(data.latitude)

    projection = ccrs.PlateCarree(central_longitude=180.0)
    fig, ax = plt.subplots(6,1,figsize=(35, 25),layout='compressed',
                            sharex=True,
                            sharey=True,
                            gridspec_kw=dict(hspace=0.03,wspace = 0.04),
                            subplot_kw={"projection": projection})
    

    # Specify contour levels and contour ticks
    cn_levels = [np.linspace(-200, 200,11),np.linspace(-200, 200,11),
                 np.linspace(-200, 200,11),np.linspace(-200, 200,11),
                 np.linspace(-200, 200,11),np.linspace(-200, 200,11)]
    lb_ticks  = cn_levels
    ####################################gv.set_vector_density

    # Create the Axes.
    for i in range(6):
        ax[i].coastlines(linewidth=0.5, zorder=1,color='gray')
        # Use geocat.viz.util convenience function to set axes tick values


        gv.set_axes_limits_and_ticks(ax[i],
                                     #xlim=[west_bd-180, east_bd-180],
                                     ylim=[south_bd, north_bd],
                                     xticks=np.arange(west_bd-180, east_bd-180+30, 30),
                                     yticks=np.arange(south_bd, north_bd+30, 30))
        # Use geocat.viz.util convenience function to add minor and major tick lines
        gv.add_major_minor_ticks(ax[i], x_minor_per_major=4,
                                      y_minor_per_major=2,
                                      labelsize=13)
        # Use geocat.viz.util convenience function to make plots look like NCL plots by
        # using latitude, longitude tick labels
        gv.add_lat_lon_ticklabels(ax[i])
        gv.set_titles_and_labels(ax[i], xlabel="Longitude",ylabel="Latitude",labelfontsize=13)
    
        # Remove the degree symbol from tick labels
        ax[i].xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
        ax[i].yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
        ax[i].tick_params(which='both', direction='in', labelsize=13)
        # Use geocat.viz.util convenience function to set titles and labels
        #gv.set_titles_and_labels(ax[i], 
        #                         maintitle='EOF'+str(i+1)+'  '+format(ev_Data1[i],'.2f')+"% (shadings) "+format(ev_Data2[i],'.2f')+"% (contours) ",
        #                         maintitlefontsize=13)
    
        shadings = ax[i].contourf(Data[i].longitude-180,Data[i].latitude,Data[i],
        levels=cn_levels[i],
        cmap=cmap,zorder=0,extend='both')

        #contour = ax[i].contour(data.longitude-180,data.latitude,Data2[i],
        #levels=cn_levels2,
        #zorder=1,colors="g",linewidths = 0.7)
#
        #ax[i].clabel(contour, levels = None, inline=True, fontsize=8)

        # Create color bars
        cbar = plt.colorbar(shadings,ax=ax[i],
                                     extendrect = True,
                                     extendfrac = 'auto',
                                     orientation='vertical',
                                     ticks=lb_ticks[i],
                                     shrink=0.9,
                                     #aspect = 40,
                                     drawedges=False)#,
                                     #pad=0.04)
        cbar.ax.tick_params(labelsize=13)
        cbar.ax.xaxis.set_label_position('bottom')
    #cbar.set_label(label=r"$\overline{\langle v^{*}' h^{*}' \rangle}$", size=13,loc = 'center')

    plt.savefig(fileout,bbox_inches='tight',pad_inches = 0.05)
    return

if __name__ == "__main__":

    years = range(2023,2024)
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    days = ["%02d" % x for x in np.arange(1,32)]
    temporal_resolution = 6
    spatial_resolution = 0.5
    fpath_in0 = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.sfc/"
    fpath_in1 = "/glade/derecho/scratch/hcluo/MSE/"
    fpath_in2 = "/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/"
    fpath_in3 = "/glade/work/hcluo/data/MSE_adjust/adjustment_"
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
            fnames4 = []
    
            year_month = str(i)+"%02d" % j
            last_day = str(calendar.monthrange(i, j)[1])
                    
            fname4 = year_month+".nc"
            fnames4.append(fpath_in3+fname4)


            fname0 = "e5.oper.an.sfc.128_134_sp.ll025sc."\
                +year_month+"0100_"+year_month+last_day+"23.nc"
            fnames0.append(fpath_in0+before(i,j)+"/"+"e5.oper.an.sfc.128_134_sp.ll025sc."\
                +before(i,j)+"0100_"+before(i,j)+last_day_before(i,j)+"23.nc")       
            fnames0.append(fpath_in0+year_month+"/"+fname0)
            fnames0.append(fpath_in0+after(i,j)+"/"+"e5.oper.an.sfc.128_134_sp.ll025sc."\
                +after(i,j)+"0100_"+after(i,j)+last_day_after(i,j)+"23.nc")

            for k in days:
                fname1 = "MSE_"+year_month+k+"00_"+year_month+k+"18.nc"
    
                fname2 = "e5.oper.an.pl.128_131_u.ll025uv." \
                +year_month+k+"00_"+year_month+k+"23.nc"
    
                fname3 = "e5.oper.an.pl.128_132_v.ll025uv." \
                +year_month+k+"00_"+year_month+k+"23.nc"
    
                folder = os.listdir(fpath_in1)
                if np.isin(fname1,folder):
                    fnames1.append(fpath_in1+fname1)
                    fnames2.append(fpath_in2+year_month+"/"+fname2)
                    fnames3.append(fpath_in2+year_month+"/"+fname3)

            MSE = read_data(fnames1,varname1,temporal_resolution,spatial_resolution)
            u = read_data(fnames2,varname2,temporal_resolution,spatial_resolution)
            v = read_data(fnames3,varname3,temporal_resolution,spatial_resolution)      
            sp = read_data(fnames0,varname0,temporal_resolution,spatial_resolution)/100.0  
            uMSE_adjust = read_data(fnames4,"uMSE",temporal_resolution,spatial_resolution)
            vMSE_adjust = read_data(fnames4,"vMSE",temporal_resolution,spatial_resolution)
          
            mask0 = maskout(MSE,sp)
            del sp
            gc.collect()
            mask = mask0.sel(time = MSE.time)

            uMSE = u*MSE
            del u
            gc.collect()
            vMSE = v*MSE
            del v
            gc.collect()
        
            uMSE_col_old = column_integrate(uMSE,mask)
            vMSE_col_old = column_integrate(vMSE,mask)
            div_H_old = divergence(uMSE_col_old,vMSE_col_old)
        
            uMSE_adj = uMSE-uMSE_adjust
            vMSE_adj = vMSE-vMSE_adjust
            del uMSE,vMSE,uMSE_adjust,vMSE_adjust
            gc.collect()
        
            uMSE_col_adj = column_integrate(uMSE_adj,mask)
            vMSE_col_adj = column_integrate(vMSE_adj,mask)
            div_H = divergence(uMSE_col_adj,vMSE_col_adj)

            F = flux(MSE,i,j)
            ddt = tendency(fnames1,file_list,MSE,varname1,mask0)

            residue = ddt+div_H-F
            residue_old = ddt+div_H_old-F

            if i == years[0] and j == months[0]:
                residue_plot = residue
                div_H_plot = div_H
                residue_old_plot = residue_old
                div_H_old_plot = div_H_old
                ddt_plot = ddt
                F_plot = F
            else:
                residue_plot = xr.concat([residue_plot,residue],dim = "time")
                div_H_plot = xr.concat([div_H_plot,div_H],dim = "time")
                residue_old_plot = xr.concat([residue_old_plot,residue_old],dim = "time")
                div_H_old_plot = xr.concat([div_H_old_plot,div_H_old],dim = "time")
                ddt_plot = xr.concat([ddt_plot,ddt],dim = "time")
                F_plot = xr.concat([F_plot,F],dim = "time")

    
    res_plot = [smooth(i) for i in [residue_plot.mean(dim = "time"),div_H_plot.mean(dim = "time"),
                                    residue_old_plot.mean(dim = "time"),div_H_old_plot.mean(dim = "time"),
                                    ddt_plot.mean(dim = "time"),F_plot.mean(dim = "time")]]

    cmap = wf.colormap(color="BluWhiRed")
    fileout = "/glade/work/hcluo/pro1/examine_1yr.pdf"
    map_plot(res_plot,cmap,fileout)
        
                  
            
    
    





