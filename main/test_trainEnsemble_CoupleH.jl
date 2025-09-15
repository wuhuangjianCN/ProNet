using CUDA,Random
Random.seed!(1234)
using ProNetM,CUDA,Dates,DataFrames,Statistics,TimeZones,JLD2,Flux,MLUtils,ProgressMeter,utilities
pInfo=getProjectInfo(dirname(pwd()));
pInfo=Base.setindex(pInfo,10000,:validLength);
ProNetM.comData=getCommenData(pInfo)
include("$(pInfo.baseFolder)/main/modelInit_EnsembleCorrector.jl");
lonM,latM=getLonLatM();
expName="Couple"
outputFolder="$(pInfo.dataFolder)/outputs/$(expName)/ensembleH";
if !isdir(outputFolder);    mkdir(outputFolder);  end
logFile="$outputFolder/log.$expName"

# ============================ get the model ==========================
poll="PM25";
yVarName=poll;
epoch_all=291:300;
xNaqVar_all=[poll,"erd_$(poll)","AER","PBLH","U","V","windSpeed","windDiv","Ter","Land","TEMP","RH","LEADTIME"];
outT=24*7;
leads_all=Hour(3) .+ Hour.(1:outT);
modelEn,normLayer,mInfo=initModel_Ensemble(poll;xNaqVar_all=xNaqVar_all,epoch_all=epoch_all,
    yVarName=yVarName,
    inT=outT,
    leads_all=leads_all,
    hid_S=[8,32,128],hid_T=256,
    nGroup_space=[1,8,16],nGroup_hid=32);
modelEn=modelEn|>gpu;
mInfo=(mInfo... ,expName=expName)
getAsample(tStart,mInfo,tLastLimit)=dataProcessor(getData_aStart(tStart,mInfo.xNaqVar_all,mInfo.yVarName,mInfo.epoch_all,mInfo.leads_all;expTag=mInfo.expName,tLastLimit=tLastLimit),mInfo.normLayer);
function getTestLoss(modelEn,tStart_allTest,mInfo)
    nBatch=length(tStart_allTest)
    loss_aEpoch=fill(NaN32,nBatch)
    num_aEpoch=fill(0,nBatch)
    tLastLimit=nothing
    tsk_getData=  Threads.@spawn getAsample(tStart_allTest[1],mInfo,tLastLimit)
    for n in eachindex(loss_aEpoch)
        data=fetch(tsk_getData) |>gpu
        count=sum(data.Y_isValid)
        if n < length(tStart_allTest)
            tsk_getData=  Threads.@spawn getAsample(tStart_allTest[n+1],mInfo,tLastLimit)
        end
        predEn=modelEn(data.X,data.ensemble)
        loss=sum((predEn - data.Y_fore).^2 .* data.Y_isValid)/count
        loss=sqrt(loss)
        loss_aEpoch[n]=loss|>cpu
        num_aEpoch[n]=count|>cpu
    end
    q_valid=num_aEpoch.>0
    weight_all=num_aEpoch ./ sum(num_aEpoch)
    loss=sum(loss_aEpoch[q_valid].*weight_all[q_valid])
    return loss
end
function getAccuracyLimit(pollName,pollLevel_all)
    concenCali=getAQIcali(pollName)
    lowerCandidate=vcat(concenCali[1:2] ./ 1.1,concenCali[3:end] ./ 1.2)
    upperCandidate=vcat(concenCali[2:end] ./ 0.8,Inf)
    lowerLimit=lowerCandidate[pollLevel_all]
    upperLimit=upperCandidate[pollLevel_all]
    return lowerLimit,upperLimit
end
# ==================================== training ===============================
function getGradient_modelE(modelEn,X,ensemble,Y_fore,weights,count)
    loss, gradsE = Flux.withgradient(modelEn) do m
        predEn=m(X,ensemble)
        loss=sum((predEn - Y_fore).^2 .* weights)/count
    end
    return loss,gradsE
end
function trainModelEn!(modelEn,opt_stateEn,tStart_aEpoch;epoch=0,mInfo=mInfo)
    Flux.trainmode!(modelEn);
    loss_aEpoch=fill(NaN32,length(tStart_aEpoch))
    num_aEpoch=fill(0,length(tStart_aEpoch))    
    tLastLimit=ZonedDateTime(2023,1,1,tz"Asia/Shanghai")
    
    tsk_Edata=  Threads.@spawn getAsample(tStart_aEpoch[1],mInfo,tLastLimit)
    p = Progress(length(tStart_aEpoch); desc="Epoch $epoch:");
    for n in eachindex(tStart_aEpoch)
        data=fetch(tsk_Edata) |>gpu
        if n < length(tStart_aEpoch)
            tsk_Edata=  Threads.@spawn getAsample(tStart_aEpoch[n+1],mInfo,tLastLimit)
        end
        count=sum(data.Y_isValid)
        weights=data.Y_isValid .* pollLevel(data.Y_fore,mInfo.yVarName)
        loss, gradsE = getGradient_modelE(modelEn,data.X,data.ensemble,data.Y_fore,weights,count)
        Flux.update!(opt_stateEn, modelEn, gradsE[1])
        loss_aEpoch[n]=loss|>cpu
        num_aEpoch[n]=count|>cpu
        data=nothing
        ProgressMeter.next!(p; showvalues=[("loss",loss)]);
    end
    q_valid=num_aEpoch.>0
    weight_all=num_aEpoch ./ sum(num_aEpoch)
    loss=sqrt(sum(loss_aEpoch[q_valid].*weight_all[q_valid]))
    return modelEn,opt_stateEn,loss
end
# constructing dataLoader ----------------
tStart_allTrain=ZonedDateTime.(DateTime(2021,12,25,12):Day(1):DateTime(2022,12,30,12),tz"UTC");
tStart_allTest=ZonedDateTime.(DateTime(2023,1,1,12):Day(20):DateTime(2023,12,31,12),tz"UTC");
# ==================================== training ===============================
data=getAsample(tStart_allTest[1],mInfo,nothing) |>gpu
count=sum(data.Y_isValid)
loss, gradsE = getGradient_modelE(modelEn,data.X,data.ensemble,data.Y_fore,data.Y_isValid,count)
function mainTrain(modelEn,tStart_allTrain,tStart_allTest,outputFolder,logFile,mInfo)
    opt_En =Adam(0.0001,(0.99, 0.9999));
    opt_stateEn = Flux.setup(opt_En, modelEn) |> gpu;
    nStart_aBatch=36;
    local model_stateEn
    for epoch in 1:1500
        GC.gc();CUDA.reclaim()
        tStart_aEpoch=rand(tStart_allTrain,nStart_aBatch)
        modelEn,opt_stateEn,loss=trainModelEn!(modelEn,opt_stateEn,tStart_aEpoch;epoch=epoch,mInfo=mInfo)
        if  rem(epoch,1)==0
            Flux.testmode!(modelEn);
            testLoss=getTestLoss(modelEn,tStart_allTest,mInfo)
            println("loss for train: ", mean(loss),"      loss for test: ", testLoss)
            # write to log file
            open(logFile,"a") do io
                println(io,"Epoch $(epoch): loss for train: ", mean(loss),"      loss for test: ", testLoss)
            end
        end
        # save model
        model_stateEn =  Flux.state(modelEn|>cpu);
        jldsave("$outputFolder/model_epoch$(epoch).jld2"; model_stateEn, opt_En,nStart_aBatch,mInfo)
    end
    return modelEn,model_stateEn
end
modelEn,model_stateEn=mainTrain(modelEn,tStart_allTrain,tStart_allTest,outputFolder,logFile,mInfo)
