using CUDA,ProNetM,Dates,DataFrames,Statistics,TimeZones,JLD2,Flux,MLUtils,ProgressMeter
pInfo=getProjectInfo(dirname(pwd()));
pInfo=Base.setindex(pInfo,16000,:validLength)
ProNetM.comData=getCommenData(pInfo)
include("$(pInfo.baseFolder)/main/modelInit_EnsembleCorrector.jl")
lonM,latM=getLonLatM()
progressBar=false
progressBar=true
# ------------------------------- prepare -------------------------------------
# prepare model ----------------------
epoch=70
expName="CoupleS"
outputFolder="$(pInfo.dataFolder)/outputs/$(expName)/ensemble"
tStart_file_all=ZonedDateTime(2021,12,24,12,tz"UTC"):Day(1):ZonedDateTime(2023,12,31,12,tz"UTC")
file_model="$(outputFolder)/model_epoch$(epoch).jld2"
model_stateEn,  mInfo=JLD2.load(file_model,"model_stateEn","mInfo") ;
yVarName=mInfo.yVarName
pollName=replace(yVarName,"erd_"=>"","_Err"=>"")
xNaqVar_all=mInfo.xNaqVar_all
epoch_all=mInfo.epoch_all
yVarName=mInfo.yVarName
leads_all=mInfo.leads_all
normLayer=mInfo.normLayer
modelEnsemble,~,~=initModel_Ensemble(pollName;xNaqVar_all=mInfo.xNaqVar_all,epoch_all=mInfo.epoch_all,
    yVarName=mInfo.yVarName,
    leads_all=mInfo.leads_all,
    hid_S=mInfo.hid_S,hid_T=mInfo.hid_T,
    nGroup_space=mInfo.nGroup_space,nGroup_hid=mInfo.nGroup_hid,
    initNorm=false
    )
modelEnsemble = Flux.testmode!( Flux.loadmodel!(modelEnsemble,model_stateEn) )|>gpu
getAsample(tStart)=dataProcessor(getData_aStart(tStart,xNaqVar_all,yVarName,epoch_all,leads_all;pInfo=pInfo,expTag=expName),normLayer)
# -------------------------------- generate the forecast ---------------------
pollName=replace(mInfo.yVarName, "erd_" => "", "_Err" => "")
if progressBar; p = Progress(length(tStart_file_all); dt=0.1); end
for tStart_file in tStart_file_all
    # calculate the AI models ----------
    data=getAsample(tStart_file) |> gpu
    predEn=dropdims(modelEnsemble(data.X,data.ensemble),dims=(3,5))
    predEn=cat(predEn,predEn[end:end,:,:];dims=1) |>cpu
    # write to files ----------------
    Times2write=tStart_file .+ mInfo.leads_all
    datFun(tStart,varName)="$outputFolder/$varName.$(Dates.format(astimezone(tStart,tz"UTC"),"yyyymmddHH")).dat"
    writeFore_aStart(predEn,tStart_file,yVarName,Times2write;datFileFun=datFun)
    # update progress bar ---------------
    if progressBar; ProgressMeter.next!(p; showvalues=[("Calculating... ",tStart_file)]); end
end
include("$(pInfo.baseFolder)/main/dataProcessing.jl")
# ----------------------------- extract Sta Data ---------------------------
fileFun_In(tStart,varName)="$outputFolder/$varName.$(Dates.format(tStart,"yyyymmddHH")).dat"
extractStaByStart(tStart_file_all,[pollName],fileFun_In;disp=true,force_overwrite=true)
# ----------------------------- combine StaData By month ---------------------
matFun_In(tStart)="$outputFolder/$pollName.$(Dates.format(tStart,"yyyymmddHH")).mat"
matFun_Month_AI(tMonth)="$outputFolder/$pollName.$(Dates.format(tMonth,"yyyymm")).mat"
mergeStaFore_byMonth(tStart_file_all,matFun_In,matFun_Month_AI)
