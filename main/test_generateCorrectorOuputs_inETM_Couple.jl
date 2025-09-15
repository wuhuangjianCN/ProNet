using CUDA
using ProNetM,CUDA,Dates,DataFrames,Statistics,TimeZones,JLD2,Flux,MLUtils,ProgressMeter,StaDataM
pInfo=getProjectInfo(dirname(pwd()));
pInfo=Base.setindex(pInfo,160000,:validLength);
ProNetM.comData=getCommenData(pInfo)
include("$(pInfo.baseFolder)/main/modelInit_Correctors_inETM.jl");
lonM,latM=getLonLatM()
progressBar=false
progressBar=true
# ------------------------------- prepare -------------------------------------
# prepare model ----------------------
for epoch in 291:300
    expTag="Couple"
    modelFolder="$(pInfo.dataFolder)/outputs/$expTag"
    outputFolder="$(pInfo.dataFolder)/outputs/$expTag/epoch$epoch"
    if !isdir(outputFolder);mkdir(outputFolder);end
    file_model="$(modelFolder)/model_epoch$(epoch).jld2"
    model_stateE,model_stateR,mInfo=JLD2.load(file_model,"model_stateE","model_stateR","mInfo") ;
    normLayer=mInfo.normLayer
    yVarName=mInfo.yVarName
    pollName=replace(yVarName,"erd_"=>"","_Err"=>"")    
    eInfo=(xVar_Einit=mInfo.xVar_Einit,xVar_Efore=mInfo.xVar_Efore,yVarName=mInfo.yVarName,
        leads2init=mInfo.leads2init,leads2Fore=mInfo.leads2Fore,pInfo=pInfo,normLayer=mInfo.normLayer)
    modelE,modelR,~,~=getModelER(;xVar_Einit=mInfo.xVar_Einit,xVar_Efore=mInfo.xVar_Efore,xVar_Rfore=mInfo.xVar_Rfore,
        yVarName=mInfo.yVarName,
        inT=mInfo.inT,outT=mInfo.outT,
        leads2init=mInfo.leads2init,leads2Fore=mInfo.leads2Fore,    
        hid_S=mInfo.hid_S,hid_T=mInfo.hid_T,hid_R=mInfo.hid_R,
        nGroup_space=mInfo.nGroup_space,nGroup_hid=mInfo.nGroup_hid,
        nGroup_R=mInfo.nGroup_R,
        initNorm=false);
    modelE = Flux.testmode!( Flux.loadmodel!(modelE,model_stateE) )|>gpu
    modelR = Flux.testmode!( Flux.loadmodel!(modelR,model_stateR) )|>gpu
    # prepare data_loader ----------------------
    tStart_file_all=ZonedDateTime(2021,12,24,12,tz"UTC"):Day(1):ZonedDateTime(2023,12,31,12,tz"UTC")
    getEdata(tStart)=dataProcessor(getEdata_aStart(tStart,eInfo.xVar_Einit,eInfo.xVar_Efore,eInfo.yVarName,eInfo.leads2init,eInfo.leads2Fore;pInfo=eInfo.pInfo),eInfo)
    getRX(tStart)=getRXmesh_aStart(tStart,mInfo.normR.mean,mInfo.normR.std;leads2Fore=mInfo.leads2Fore,xVar=mInfo.xVar_Rfore)
    tsk_Edata=  Threads.@spawn getEdata(tStart_file_all[1])
    tsk_Rdata=  Threads.@spawn getRX(tStart_file_all[1]);
    # -------------------------------- generate the forecast ---------------------
    if progressBar; p = Progress(length(tStart_file_all); dt=0.1, desc="Epoch $epoch:"); end
    for n_start in eachindex(tStart_file_all)
        tStart_file=tStart_file_all[n_start]
        # prepare data ------------------------
        dataE=fetch(tsk_Edata)
        RXmesh=fetch(tsk_Rdata)
        if n_start < length(tStart_file_all)
            tsk_Edata=  Threads.@spawn getEdata(tStart_file_all[n_start+1])
            tsk_Rdata=Threads.@spawn getRX(tStart_file_all[n_start+1])
        end
        # calculate modelE -------------
        dataGPU=(X_Einit=dataE.X_Einit,X_Efore=dataE.X_Efore) |>gpu
        predE=modelE(dataGPU.X_Einit,dataGPU.X_Efore)
        # calculate modelR -------------
        X_allValid=getValidSample(RXmesh,dataE.Y_isValid)
        predR=hcat(cpu(collect(modelR(X) for X in gpu(chunk(X_allValid,5))))...)
        predRmesh=predE .* 0.0;predRmesh[dataE.Y_isValid] =predR
        # calculate the process-driven forecasting ----------
        toFileFormat(x)=dropdims(cat(x,x[end:end,:,:,:,:],dims=1),dims=(3,5))
        predE=toFileFormat(predE|>cpu)
        predRmesh=toFileFormat(predRmesh|>cpu)
        # write to files ----------------
        Times2write=tStart_file .+ mInfo.leads2Fore
        datFun(tStart,varName)="$outputFolder/$varName.$(Dates.format(astimezone(tStart,tz"UTC"),"yyyymmddHH")).dat"
        writeFore_aStart(predE,tStart_file,"$(pollName)_predE",Times2write;datFileFun=datFun)
        writeFore_aStart(predRmesh,tStart_file,"$(pollName)_predR",Times2write;datFileFun=datFun)
        # update progress bar ---------------
        if progressBar; ProgressMeter.next!(p; showvalues=[("Calculating... ",tStart_file)]); end
    end
end
