using ProNetM,Dates,DataFrames,Statistics,TimeZones,MyPlotM,JLD2,Flux,MLUtils,ProgressMeter
pInfo=getProjectInfo(dirname(pwd()));
pInfo=Base.setindex(pInfo,160000,:validLength)
ProNetM.comData=getCommenData(pInfo)
lonM,latM=getLonLatM()
pollName="PM25"
progressBar=false
progressBar=true
include("$(pInfo.baseFolder)/main/dataProcessing.jl")
for epoch in 101:110
    expTag="CoupleS"
    outputFolder="$(pInfo.dataFolder)/outputs/$expTag/epoch$epoch"
    # prepare data_loader ----------------------
    tStart_file_all=ZonedDateTime(2021,12,24,12,tz"UTC"):Day(1):ZonedDateTime(2023,12,31,12,tz"UTC")
    # -------------------------------- generate the forecast ---------------------
    if progressBar; p = Progress(length(tStart_file_all); dt=0.1, desc="Epoch $epoch:"); end
    Threads.@threads for n_t in eachindex(tStart_file_all)
        tStart_file=tStart_file_all[n_t]
        Times_aStart=tStart_file .+ (Hour(3):Hour(1):Hour(24*7+3))
        # read predE and dRatio -------------------
        datFun_ProNet(tStart,varName)="$outputFolder/$varName.$(Dates.format(astimezone(tStart,tz"UTC"),"yyyymmddHH")).dat"
        varNameS=["$(pollName)_predE","$(pollName)_predR",]
        data_aStart=readFore_aStart(tStart_file,varNameS,Times_aStart;grdIn=copy(pInfo.grdNaq), datFileFun=datFun_ProNet)
        predE=data_aStart[:,:,1,:]
        predR=data_aStart[:,:,2,:]
        # read U_all, V_all, poll_anaInit
        data_aStart=readFore_aStart(tStart_file,[pollName,"U","V"],Times_aStart;grdIn=copy(pInfo.grdNaq))
        OriFore_all=data_aStart[:,:,1,:]
        U_all=data_aStart[:,:,2,:]
        V_all=data_aStart[:,:,3,:] 
        poll_anaInit=dropdims(readAnalysis_aStart(tStart_file,pollName,[Times_aStart[1]];grdre=copy(pInfo.grdRe)),dims=(3,4))
        q_validGrid=dropdims(readValidGrid(tStart_file,pollName,Times_aStart;grdre=copy(pInfo.grdRe)),dims=3)
        # calculate the process-driven forecasting ----------    
        erdErr2add=predE + predR
        erdErr2add[isnan.(erdErr2add)].=0
        err_init=OriFore_all[:,:,1] - poll_anaInit
        newFore_all,erdErr_all,advErr_all=ETM(OriFore_all,U_all,V_all,err_init;erdErr2add=erdErr2add, 
            dx=15e3,dy=15e3,par=false,non_negative=true,max_con=nothing,analysis=true)
        # write to files ----------------
        writeFore_aStart(erdErr_all,tStart_file,"erd_$(pollName)_Err",Times_aStart;datFileFun=datFun_ProNet,grdOut=copy(pInfo.grdNaq))
        writeFore_aStart(newFore_all,tStart_file,pollName,Times_aStart;datFileFun=datFun_ProNet,grdOut=copy(pInfo.grdNaq))
        # update progress bar ---------------
        if progressBar; ProgressMeter.next!(p; showvalues=[("Calculating... ",tStart_file)]); end
    end
    # ----------------------------- extract Sta Data ---------------------------
    fileFun_In(tStart,varName)="$(pInfo.dataFolder)/outputs/$expTag/epoch$epoch/$varName.$(Dates.format(tStart,"yyyymmddHH")).dat"
    extractStaByStart(tStart_file_all,[pollName],fileFun_In;disp=true,force_overwrite=true)
    # ----------------------------- combine StaData By month ---------------------
    matFun_In(tStart)="$(pInfo.dataFolder)/outputs/$expTag/epoch$epoch/$pollName.$(Dates.format(tStart,"yyyymmddHH")).mat"
    matFun_Month_AI(tMonth)="$(pInfo.dataFolder)/outputs/$expTag/epoch$epoch/$pollName.$(Dates.format(tMonth,"yyyymm")).mat"
    mergeStaFore_byMonth(tStart_file_all,matFun_In,matFun_Month_AI)
end