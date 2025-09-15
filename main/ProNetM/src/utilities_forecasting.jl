using NaqpmsUtilities,NaNStatistics,Dates,TimeZones
export ETM,IETM,calForeRMSE,calForeWithERD,generateForecastWithErd
# calculate the pronet forecast
function ETM(OriFore_all::AbstractArray{T},U_all,V_all,err_init;erdErr2add=nothing,erdErr2ratio=nothing,
    depRatioOri=nothing,depRatioPro=nothing,dx=15e3,dy=15e3,par=false,non_negative=true,max_con=nothing,analysis=false) where T
    err_all=similar(OriFore_all)
    erdErr_all=zeros(size(OriFore_all))
    advErr_all=zeros(size(OriFore_all))
    newFore_all=similar(OriFore_all)
    err_all[:,:,1]=err_init
    newFore_all[:,:,1]=OriFore_all[:,:,1] - err_init

    max_con= isnothing(max_con) ? 1e2 * maximum(OriFore_all) : max_con
    DX=fill(convert(T,dx),size(U_all)[1:2])
    DY=fill(convert(T,dy),size(U_all)[1:2])
    for n_t in axes(U_all,3)[2:end]
        # println(n_t)
        advErr_all[:,:,n_t] =  advDiffAnHour(err_all[:,:,n_t-1], U_all[:,:,n_t], V_all[:,:,n_t],  DX, DY;par=par,num_tStep=NaN)
        if !isnothing(erdErr2ratio) # correct deposition
            erdErr_all[:,:,n_t] .+= (OriFore_all[:,:,n_t] - advErr_all[:,:,n_t] ) .* erdErr2ratio[:,:,n_t]
        end
        if !isnothing(erdErr2add) # correct emission
            erdErr_all[:,:,n_t] += erdErr2add[:,:,n_t]
        end
        newFore_all[:,:,n_t] = OriFore_all[:,:,n_t] -  advErr_all[:,:,n_t] - erdErr_all[:,:,n_t]
        if !isnothing(depRatioOri) && !isnothing(depRatioPro)
            depErr= depRatioPro[:,:,n_t] .* newFore_all[:,:,n_t]  -  depRatioOri[:,:,n_t] .*  OriFore_all[:,:,n_t]
            erdErr_all[:,:,n_t] += depErr
            newFore_all[:,:,n_t] = OriFore_all[:,:,n_t] -  advErr_all[:,:,n_t] - erdErr_all[:,:,n_t]
        end
        if non_negative # if force positive concentration
            newFore_all[:,:,n_t] = max.(newFore_all[:,:,n_t],0)
        end
        newFore_all[:,:,n_t] = min.(newFore_all[:,:,n_t],max_con)
        err_all[:,:,n_t] = OriFore_all[:,:,n_t] - newFore_all[:,:,n_t]
    end
    if analysis 
        return newFore_all,erdErr_all,advErr_all
    else
        return newFore_all
    end
end
function IETM(err_init,U_all::AbstractArray{T},V_all;erdErr=nothing,dx=15e3,dy=15e3,par=false) where T
    err_all=similar(U_all)
    err_all[:,:,1]=err_init
    DX=fill(convert(T,dx),size(U_all)[1:2])
    DY=fill(convert(T,dy),size(U_all)[1:2])
    for n_t in axes(U_all,3)[2:end]
        err_all[:,:,n_t] =  advDiffAnHour(err_all[:,:,n_t-1], U_all[:,:,n_t], V_all[:,:,n_t],  DX, DY;par=par,num_tStep=NaN)
        if !isnothing(erdErr)
            err_all[:,:,n_t] .+= erdErr[:,:,n_t]
        end
    end
    return err_all
end
function calForeRMSE(fore,ana,q_validGrid)
    err=fore-ana
    err[.!q_validGrid].=NaN32
    out_rmse=vec(sqrt.(nanmean(err.^2;dims=(1,2))))
    return out_rmse
end
function calForeWithERD(tStart::ZonedDateTime,pollName::AbstractString,erdErr2add,erdErr2ratio;OriFore_all=nothing,U_all=nothing,V_all=nothing,q_validGrid=nothing,par=true,non_negative=true,analysis=false)
    
    Times=tStart .+ Hour.(0:176)
    OriFore_all= isnothing(OriFore_all) ? dropdims(readFore_aStart(tStart,pollName,Times),dims=3) : OriFore_all
    U_all= isnothing(U_all) ? dropdims(readFore_aStart(tStart,"U",Times),dims=3) : U_all
    V_all= isnothing(V_all) ? dropdims(readFore_aStart(tStart,"V",Times),dims=3) : V_all
    q_validGrid= isnothing(q_validGrid) ? dropdims(readValidGrid_aStart(tStart,pollName,Times),dims=3) : q_validGrid
    
    erdErr2add[.!q_validGrid] .= 0.0
    erdErr2ratio[.!q_validGrid] .= 0.0
    poll_anaInit=dropdims(readAnalysis_aStart(tStart,pollName,[tStart]),dims=(3,4))
    err_init=OriFore_all[:,:,1]-poll_anaInit
    # println("nan in OriFore_all: ",sum(isnan.(OriFore_all)),sum(isnan.(U_all)),sum(isnan.(V_all)),sum(isnan.(q_validGrid)),sum(isnan.(err_init)))
    if analysis
        newFore_all,erdErr_all=ETM(OriFore_all,U_all,V_all,err_init;erdErr2add=erdErr2add,erdErr2ratio=erdErr2ratio,dx=15e3,dy=15e3,par=par,non_negative=non_negative,analysis=analysis) 
        return newFore_all,OriFore_all,erdErr_all,U_all,V_all
    else
        newFore_all=ETM(OriFore_all,U_all,V_all,err_init;erdErr2add=erdErr2add,erdErr2ratio=erdErr2ratio,dx=15e3,dy=15e3,par=par,non_negative=non_negative,analysis=analysis) 
        return newFore_all
    end
end
function calForeWithERD(tStart_file::ZonedDateTime,tStart_curr::ZonedDateTime,pollName::AbstractString,erdErr2add,erdErr2ratio;OriFore_all=nothing,U_all=nothing,V_all=nothing,q_validGrid=nothing,poll_anaInit=nothing,par=true,non_negative=true,analysis=false)
    
    Times=tStart_curr .+ Hour.(0:size(erdErr2add,3)-1)
    OriFore_all= isnothing(OriFore_all) ? dropdims(readFore_aStart(tStart_file,pollName,Times),dims=3) : OriFore_all
    U_all= isnothing(U_all) ? dropdims(readFore_aStart(tStart_file,"U",Times),dims=3) : U_all
    V_all= isnothing(V_all) ? dropdims(readFore_aStart(tStart_file,"V",Times),dims=3) : V_all
    q_validGrid= isnothing(q_validGrid) ? dropdims(readValidGrid_aStart(tStart_file,pollName,Times),dims=3) : q_validGrid

    erdErr2add[.!q_validGrid] .= 0.0
    erdErr2ratio[.!q_validGrid] .= 0.0
    poll_anaInit=isnothing(poll_anaInit) ? dropdims(readAnalysis_aStart(tStart_file,pollName,[tStart_curr]),dims=(3,4)) : poll_anaInit
    err_init=OriFore_all[:,:,1]-poll_anaInit
    # println("nan in OriFore_all: ",sum(isnan.(OriFore_all)),sum(isnan.(U_all)),sum(isnan.(V_all)),sum(isnan.(q_validGrid)),sum(isnan.(err_init)))
    if analysis
        newFore_all,erdErr_all=ETM(OriFore_all,U_all,V_all,err_init;erdErr2add=erdErr2add,erdErr2ratio=erdErr2ratio,dx=15e3,dy=15e3,par=par,non_negative=non_negative,analysis=analysis) 
        return newFore_all,OriFore_all,erdErr_all,U_all,V_all
    else
        newFore_all=ETM(OriFore_all,U_all,V_all,err_init;erdErr2add=erdErr2add,erdErr2ratio=erdErr2ratio,dx=15e3,dy=15e3,par=par,non_negative=non_negative,analysis=analysis) 
        return newFore_all
    end
end
function calForeWithERD(tStart_file::ZonedDateTime,tStart_curr::ZonedDateTime,pollName::AbstractString,erd_all;OriFore_all=nothing,U_all=nothing,V_all=nothing,q_validGrid=nothing,poll_anaInit=nothing,par=true,non_negative=true,analysis=false)
    
    Times=tStart_curr .+ Hour.(0:size(erd_all,3)-1)
    OriFore_all= isnothing(OriFore_all) ? dropdims(readFore_aStart(tStart_file,pollName,Times),dims=3) : OriFore_all
    U_all= isnothing(U_all) ? dropdims(readFore_aStart(tStart_file,"U",Times),dims=3) : U_all
    V_all= isnothing(V_all) ? dropdims(readFore_aStart(tStart_file,"V",Times),dims=3) : V_all
    q_validGrid= isnothing(q_validGrid) ? dropdims(readValidGrid_aStart(tStart_file,pollName,Times),dims=3) : q_validGrid

    erd_all[.!q_validGrid] .= 0.0
    poll_anaInit=isnothing(poll_anaInit) ? dropdims(readAnalysis_aStart(tStart_file,pollName,[tStart_curr]),dims=(3,4)) : poll_anaInit
    err_init=OriFore_all[:,:,1]-poll_anaInit    
    if analysis
        newFore_all,erdErr_all=ETM(OriFore_all,U_all,V_all,err_init;erdErr2add=erd_all,dx=15e3,dy=15e3,par=par,non_negative=non_negative,analysis=analysis) 
        return newFore_all,OriFore_all,erdErr_all,U_all,V_all
    else
        newFore_all=ETM(OriFore_all,U_all,V_all,err_init;erdErr2add=erd_all,dx=15e3,dy=15e3,par=par,non_negative=non_negative,analysis=analysis) 
        return newFore_all
    end
end
function calForeWithERD(tStart::ZonedDateTime,pollName::AbstractString,erd_all;OriFore_all=nothing,U_all=nothing,V_all=nothing,q_validGrid=nothing,par=true,non_negative=true)
    
    Times=tStart .+ Hour.(0:176)
    OriFore_all= isnothing(OriFore_all) ? dropdims(readFore_aStart(tStart,pollName,Times),dims=3) : OriFore_all
    U_all= isnothing(U_all) ? dropdims(readFore_aStart(tStart,"U",Times),dims=3) : U_all
    V_all= isnothing(V_all) ? dropdims(readFore_aStart(tStart,"V",Times),dims=3) : V_all
    q_validGrid= isnothing(q_validGrid) ? dropdims(readValidGrid_aStart(tStart,pollName,Times),dims=3) : q_validGrid

    erd_all[.!q_validGrid] .= 0.0
    poll_anaInit=dropdims(readAnalysis_aStart(tStart,pollName,[tStart]),dims=(3,4))
    err_init=OriFore_all[:,:,1]-poll_anaInit
    newForecast=ETM(OriFore_all,U_all,V_all,err_init;erdErr2add=erd_all,dx=15e3,dy=15e3,par=par,non_negative=non_negative) 
    return newForecast
end
function generateForecastWithErd(tStart::ZonedDateTime,pollName::AbstractString,outputFolder::AbstractString;par=true)
    Times=tStart .+ Hour.(0:176)
    datFun(tStart,varName)="$outputFolder/$varName.$(Dates.format(astimezone(tStart,tz"UTC"),"yyyymmddHH")).dat"
    erd_all=dropdims(readFore_aStart(tStart,"erd_"*pollName*"_Err",Times;datFileFun=datFun),dims=3)
    newFore=calForeWithERD(tStart,pollName,erd_all;par=par)
    writeFore_aStart(newFore,tStart,pollName,Times;datFileFun=datFun)
    return newFore
end

function generateForecastWithErd(tStart_file::ZonedDateTime,tStart_curr::ZonedDateTime,pollName::AbstractString,outputFolder::AbstractString;par=true,tEnd_file=tStart_file +Hour(176))
    Times=tStart_curr:Hour(1):tEnd_file
    datFun(tStart,varName)="$outputFolder/$varName.$(Dates.format(astimezone(tStart,tz"UTC"),"yyyymmddHH")).dat"
    erd_all=dropdims(readFore_aStart(tStart_file,"erd_"*pollName*"_Err",Times;datFileFun=datFun),dims=3)
    newFore=calForeWithERD(tStart_file,tStart_curr,pollName,erd_all;par=par)
    writeFore_aStart(newFore,tStart_file,pollName,Times;datFileFun=datFun)
    return newFore
end

function generateForecastWithErd(tStart::ZonedDateTime,pollName::AbstractString,erdErr::AbstractArray;par=true)
    dim2drop=Tuple(findall(size(erdErr) .== 1))
    erdErr=dropdims(erdErr,dims=dim2drop)
    newFore=calForeWithERD(tStart,pollName,erdErr;par=par)
    return newFore
end


raw"""
using Revise,Plots,NaqpmsUtilities,Statistics,Dates,TimeZones,NaNStatistics
# plot results
include("playground/myDataLoader.jl")
tStart=ZonedDateTime(2022,1,2,12,tz"UTC")
Times=tStart .+ Hour.(0:168)
pm25=dropdims(readFore_aStart(tStart,"PM25",Times),dims=3)  
U_all=dropdims(readFore_aStart(tStart,"U",Times),dims=3)
V_all=dropdims(readFore_aStart(tStart,"V",Times),dims=3)  
pm25_ana=dropdims(readAnalysis(Times,"PM25"),dims=3)  
q_validGrid=dropdims(readValidGrid(Times,"PM25"),dims=3)  
erdFileFun(tStartStr,varName)="$baseFolder/DATA/outputs/expBase/$varName.$tStartStr.dat"
pm25_erd_pred=dropdims(readFore_aStart(tStart,"erd_PM25_Err",Times;datFileFun=erdFileFun),dims=3)
err_init=pm25[:,:,1]-pm25_ana[:,:,1]
pm25_ietm=pm25 - IETM(err_init,U_all,V_all;par=true);
pm25_pronet=pm25 - IETM(err_init,U_all,V_all;erdErr=pm25_erd_pred,par=true);
data2plot=hcat(calForeRMSE(pm25,pm25_ana,q_validGrid),calForeRMSE(pm25_ietm,pm25_ana,q_validGrid),calForeRMSE(pm25_pronet,pm25_ana,q_validGrid))

plot(0:length(Times)-1,data2plot,label=["Naqpms" "IETM" "ProNet"],lw=3,legend=:topleft)
xticks!(0:24:length(Times))
"""