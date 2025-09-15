using CUDA,Flux,Revise,ProNetM,Dates,DataFrames,Statistics,TimeZones,MLUtils,ProgressMeter,JLD2,TimeZones
pInfo=getProjectInfo(dirname(pwd()));
pInfo=Base.setindex(pInfo,160000,:validLength)
ProNetM.comData=getCommenData(pInfo)
includet("$(pInfo.baseFolder)/main/modelInit_PureAI.jl");
includet("$(pInfo.baseFolder)/plots/dataProcessing.jl")
lonM,latM=getLonLatM()
progressBar=false
progressBar=true
# ------------------------------- prepare -------------------------------------
# prepare model ----------------------
for epoch in 191:200
    expTag="PureAI"
    modelFolder="$(pInfo.dataFolder)/outputs/$expTag"
    outputFolder="$(pInfo.dataFolder)/outputs/$expTag/epoch$epoch"
    if !isdir(outputFolder);mkdir(outputFolder);end
    file_model="$(modelFolder)/model_epoch$(epoch).jld2"
    model_stateE,mInfo=JLD2.load(file_model,"model_stateE","mInfo") ;
    normLayer=mInfo.normLayer
    yVarName=mInfo.yVarName
    pollName=replace(yVarName,"erd_"=>"","_Err"=>"")
    modelE,~,~=getModelST(;xVar_Einit=mInfo.xVar_Einit, xVar_Efore=mInfo.xVar_Efore,
        yVarName=mInfo.yVarName,
        inT=mInfo.inT,outT=mInfo.outT,
        leads2init=mInfo.leads2init,leads2Fore=mInfo.leads2Fore,
        hid_S=mInfo.hid_S,hid_T=mInfo.hid_T,
        initNorm=false)
    modelE = Flux.testmode!( Flux.loadmodel!(modelE,model_stateE) )|>gpu
    # prepare data_loader ----------------------
    tStart_file_all=ZonedDateTime(2021,12,24,12,tz"UTC"):Day(1):ZonedDateTime(2023,12,31,12,tz"UTC")
    getAsample(tStart)=dataProcessor(getData_aStart(tStart,mInfo.xVar_Einit,mInfo.xVar_Efore,
        mInfo.yVarName,mInfo.leads2init,mInfo.leads2Fore;pInfo=pInfo),mInfo)
    # -------------------------------- generate the forecast ---------------------
    if progressBar; p = Progress(length(tStart_file_all); dt=0.1, desc="Epoch $epoch:"); end
    for (n_start,tStart_file) in enumerate(tStart_file_all)
        # calculate the AI models ----------
        data_Curr=getAsample(tStart_file) |> gpu
        predE=modelE(data_Curr.X_Einit,data_Curr.X_Efore)
        toFileFormat(x)=dropdims(cat(x,x[end:end,:,:,:,:],dims=1),dims=(3,5))
        # calculate the process-driven forecasting ----------
        predE=toFileFormat(predE)|>cpu
        # write forecasting error to files ----------------
        Times2write=tStart_file .+ mInfo.leads2Fore
        datFun(tStart,varName)="$outputFolder/$varName.$(Dates.format(astimezone(tStart,tz"UTC"),"yyyymmddHH")).dat"
        writeFore_aStart(predE,tStart_file,yVarName,Times2write;datFileFun=datFun)
        # update progress bar ---------------
        if progressBar; ProgressMeter.next!(p; showvalues=[("Calculating... ",tStart_file)]); end
    end
end
