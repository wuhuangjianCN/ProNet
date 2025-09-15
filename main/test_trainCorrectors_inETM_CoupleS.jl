using CUDA,Random
Random.seed!(1234)
using ProNetM,Dates,DataFrames,Statistics,TimeZones,JLD2,Flux,MLUtils,ProgressMeter,StaDataM
pInfo=getProjectInfo(dirname(pwd()));
pInfo=Base.setindex(pInfo,30000,:validLength);
ProNetM.comData=getCommenData(pInfo)
include("$(pInfo.baseFolder)/main/modelInit_Correctors_inETM.jl");
lonM,latM=getLonLatM();
expName="CoupleS";
outputFolder="$(pInfo.dataFolder)/outputs/$(expName)";
if !isdir(outputFolder);    mkdir(outputFolder);  end;
logFile="$outputFolder/log.$expName";

progressBar=true;
dispAna=false;
# ============================ get the model ==========================
poll="PM25";
xVar_Einit=[poll,"erd_$(poll)","erd_$(poll)_Err","$(poll)_Err","AER","erd_AER","PBLH","U","V","windSpeed","windDiv","Ter","Land","TEMP","RH"];
xVar_Efore=[poll,"erd_$(poll)","AER","erd_AER","PBLH","U","V","windSpeed","windDiv","Ter","Land","TEMP","RH"];
xVar_Rfore=[poll,"erd_$(poll)","erd_AER","erd_ANH4", "erd_ANO3",  "erd_HONO", "erd_ASO4", "erd_NH3", "erd_OC", "erd_HCHO",  "erd_CO", "erd_NO2",
    "erd_SOA", "erd_NO", "erd_C2H6", "erd_SO2","erd_ISOP", "erd_O3", "erd_HNO3", "TEMP", "RH"];
yVarName="erd_$(poll)_Err";

inT=24*7;outT=24*7;
leads2init= Hour(3) .+ Hour.(-(inT-1):0);
leads2Fore=Hour(3) .+ Hour.(1:outT);
modelE,modelR,normLayer,mInfo=getModelER(;xVar_Einit=xVar_Einit,xVar_Efore=xVar_Efore,xVar_Rfore=xVar_Rfore,
    yVarName=yVarName,
    inT=inT,outT=outT,
    leads2init=leads2init,leads2Fore=leads2Fore,
    hid_S=[8,16,128],hid_T=256,hid_R=1024,
    nGroup_space=[1,8,16],nGroup_hid=32,
    Rdrop_ratio=[0.1,0.1,0.1] .* 2,
    initNorm=true);

modelE=modelE |> gpu ;
modelR=modelR |> gpu;
function getTestLoss(modelE,modelR,testE_looper;progressBar=true,mInfo=mInfo,calModelR=true)  
    nBatch=length(testE_looper.idx_all)
    loss_aEpoch=fill(NaN32,nBatch,3)
    num_aEpoch=fill(0,nBatch)
    reset!(testE_looper)
    tStart_file_all=testE_looper.tStart_all[testE_looper.idx_all]
    tsk_Edata= Threads.@spawn next(testE_looper);
    Flux.testmode!(modelE);
    getRX(tStart)=getRXmesh_aStart(tStart,mInfo.normR.mean,mInfo.normR.std;leads2Fore=mInfo.leads2Fore,xVar=mInfo.xVar_Rfore)
    if calModelR; tsk_Rdata=  Threads.@spawn getRX(tStart_file_all[1]); end
    if progressBar; p = Progress(1+nBatch; desc="Testing:",dt=0.1); end
    for n in 1:nBatch
        tStart_curr=tStart_file_all[n]
        # ------------------ for model E ------------------------
        data=fetch(tsk_Edata)
        if tStart_curr != testE_looper.tReaded
            @error "tSTart for modelE and modelR is different!!!!!!!!!!!!!!"
        end
        if n < length(loss_aEpoch)
            tsk_Edata= Threads.@spawn next(testE_looper);
        end
        dataGPU=(X_Einit=data.X_Einit,X_Efore=data.X_Efore,Y_fore=data.Y_fore,Y_isValid=data.Y_isValid) |>gpu
        count=sum(dataGPU.Y_isValid)
        predE=modelE(dataGPU.X_Einit,dataGPU.X_Efore)        
        # --------------------- for model R --------------------
        if calModelR
            RXmesh=fetch(tsk_Rdata)
            if n < length(tStart_file_all)
                tsk_Rdata=Threads.@spawn getRX(tStart_file_all[n+1])
            end
            X_allValid=getValidSample(RXmesh,data.Y_isValid)
            predR=hcat(cpu(collect(modelR(X) for X in gpu(chunk(X_allValid,5))))...)
            predRmesh=predE .* 0.0;predRmesh[data.Y_isValid] =predR
            predRmesh=predRmesh|>gpu
        else
            predRmesh= predE .* 0.0 |>gpu
        end
        # ----------------- loss an others --------------------------
        loss=[sum(abs.(dataGPU.Y_fore) .* dataGPU.Y_isValid)/count,
            sum(abs.(predE - dataGPU.Y_fore) .* dataGPU.Y_isValid)/count,
            sum(abs.(predE + predRmesh - dataGPU.Y_fore) .* dataGPU.Y_isValid)/count]
        loss_aEpoch[n,:] = loss|>cpu
        num_aEpoch[n]=count|>cpu
        if progressBar; ProgressMeter.next!(p;showvalues=[("loss",loss)]); end
    end
    q_valid=num_aEpoch.>0
    weight_all=num_aEpoch ./ sum(num_aEpoch)
    loss=sum(loss_aEpoch[q_valid,:].*weight_all[q_valid],dims=1)
    if progressBar; ProgressMeter.next!(p; showvalues=[("loss for Test:",loss)]); end
    return loss
end

# ==================================== training ===============================
function getGradient_modelE(modelE,X_Einit,X_Efore,Y_fore,targetY_curr,Y_isValid,count)
    loss, gradsE = Flux.withgradient(modelE) do m
        predE=m(X_Einit,X_Efore)
        loss=sum((abs.(predE  - targetY_curr) ) .* Y_isValid)/count + sum((abs.(predE  - Y_fore) ) .* Y_isValid)/count
    end
    return loss,gradsE
end
function trainModelE!(modelE,modelR,opt_stateE,eInfo,tStart_aEpoch;epoch=0,calModelR=false,mInfo=mInfo)
    Flux.trainmode!(modelE);Flux.testmode!(modelR);
    loss_aEpoch=fill(NaN32,length(tStart_aEpoch))
    num_aEpoch=fill(0,length(tStart_aEpoch))        
    tLastLimit=ZonedDateTime(2023,1,1,tz"Asia/Shanghai")
    getRX(tStart)=getRXmesh_aStart(tStart,mInfo.normR.mean,mInfo.normR.std;leads2Fore=mInfo.leads2Fore,xVar=mInfo.xVar_Rfore)
    getEdata(tStart)=dataProcessor(getEdata_aStart(tStart,eInfo.xVar_Einit,eInfo.xVar_Efore,eInfo.yVarName,eInfo.leads2init,eInfo.leads2Fore;pInfo=eInfo.pInfo,tLastLimit=tLastLimit),eInfo)
    if calModelR; tsk_Rdata=  Threads.@spawn getRX(tStart_aEpoch[1]); end
    tsk_Edata=  Threads.@spawn getEdata(tStart_aEpoch[1])
    p = Progress(length(tStart_aEpoch); desc="Epoch $epoch:");
    for n in eachindex(tStart_aEpoch)
        data=fetch(tsk_Edata)
        if n < length(tStart_aEpoch)
            tsk_Edata=  Threads.@spawn getEdata(tStart_aEpoch[n+1])
        end
        X_Einit=data.X_Einit |>gpu
        X_Efore=data.X_Efore |>gpu
        Y_isValid_cpu=data.Y_isValid
        Y_isValid=Y_isValid_cpu |>gpu
        Y_fore=data.Y_fore |>gpu
        count=sum(Y_isValid)
        if calModelR
            RXmesh=fetch(tsk_Rdata)
            if n < length(tStart_aEpoch)
                tsk_Rdata=Threads.@spawn getRX(tStart_aEpoch[n+1])
            end
            X_allValid=getValidSample(RXmesh,data.Y_isValid)
            predR=hcat(cpu(collect(modelR(X) for X in gpu(chunk(X_allValid,5))))...)
            predRmesh=Y_fore .* 0.0;predRmesh[data.Y_isValid] =predR
            predRmesh=predRmesh|>gpu
            targetY_curr=Y_fore - predRmesh
        else
            targetY_curr=Y_fore
        end
        loss, gradsE = getGradient_modelE(modelE,X_Einit,X_Efore,Y_fore,targetY_curr,Y_isValid,count)
        Flux.update!(opt_stateE, modelE, gradsE[1])
        loss_aEpoch[n]=loss|>cpu
        num_aEpoch[n]=count|>cpu
        ProgressMeter.next!(p; showvalues=[("loss",loss)]);
    end
    q_valid=num_aEpoch.>0
    weight_all=num_aEpoch ./ sum(num_aEpoch)
    loss=sum(loss_aEpoch[q_valid].*weight_all[q_valid])
    return modelE,opt_stateE,loss
end

function trainModelR!(modelE,modelR,opt_stateR,eInfo,mInfo,tStart_aEpoch;progressBar=progressBar,nRChunk_aEpoch=36)
    if progressBar; p = Progress(length(tStart_aEpoch)+2; desc="Rtrianing:",dt=0.1); end
    trainX,trainY=getRXTraining(tStart_aEpoch,modelE,mInfo,eInfo;progressBar=false)
    count_all=size(trainY,2)
    batchSize=Int(ceil(length(trainY)/nRChunk_aEpoch))
    Rtrain_loader=Flux.DataLoader((trainX|>gpu,trainY|>gpu); batchsize=batchSize, shuffle=true, partial=true)
    Flux.trainmode!(modelR);
    loss_aEpoch,num_aEpoch=[],[]
    nRChunk=0
    local gradsR
    if progressBar;  ProgressMeter.next!(p); end
    for (X,Y) in Rtrain_loader
        nRChunk+=1
        X=X|>gpu;
        targetY=Y[1:1,:]|>gpu;
        Y_fore=Y[2:2,:]|>gpu;
        count_curr=length(targetY)
        loss, gradsR_curr = Flux.withgradient(modelR) do m
            predR=m(X)
            loss=mean(
                abs.(predR  - targetY)  +
                abs.(predR  - Y_fore)  +   Float32(0.05) .* abs.(predR)       )
        end        
        # update gradsD
        if nRChunk == 1
            gradsR=fmap(x -> isnothing(x) ? nothing : (count_curr/count_all) .* x, gradsR_curr)
        else
            gradsR=fmap((x,y)-> isnothing(y) ? nothing : x + (count_curr/count_all) .* y, gradsR,gradsR_curr)
        end
        # update figure ------
        loss_aEpoch=push!(loss_aEpoch,loss|>cpu)
        num_aEpoch=push!(num_aEpoch,length(Y)|>cpu)
        if progressBar;  ProgressMeter.next!(p; showvalues=[("loss",loss)]); end
        if nRChunk >= nRChunk_aEpoch; break; end
    end
    Flux.update!(opt_stateR, modelR, gradsR[1])
    weight_all=num_aEpoch ./ sum(num_aEpoch)
    trainLoss=sum(loss_aEpoch.*weight_all)
    if progressBar;  ProgressMeter.next!(p; showvalues=[("loss",trainLoss)]); end
    return modelR,opt_stateR,trainLoss
end

# ----------------------------------- prepare data for training modelR ------------------------
function getRXTraining(tStart_aEpoch,modelE,mInfo,eInfo;progressBar=true)
    X_allStart=Vector{Any}(undef,length(tStart_aEpoch))
    Y_allStart=Vector{Any}(undef,length(tStart_aEpoch))
    tLastLimit=ZonedDateTime(2023,1,1,tz"Asia/Shanghai")
    Flux.testmode!(modelE);
    getRX(tStart)=getRXmesh_aStart(tStart,mInfo.normR.mean,mInfo.normR.std;leads2Fore=mInfo.leads2Fore[1:24],xVar=mInfo.xVar_Rfore)
    getEdata(tStart)=dataProcessor(getEdata_aStart(tStart,eInfo.xVar_Einit,eInfo.xVar_Efore,eInfo.yVarName,eInfo.leads2init,eInfo.leads2Fore;pInfo=eInfo.pInfo,tLastLimit=tLastLimit),eInfo)
    tsk_Rdata=  Threads.@spawn getRX(tStart_aEpoch[1])
    tsk_Edata=  Threads.@spawn getEdata(tStart_aEpoch[1])
    if progressBar; p = Progress(length(tStart_aEpoch); desc="getRXdata:",dt=0.1); end
    for n in eachindex(tStart_aEpoch)
        # ------------------ for RY ------------------------
        data=fetch(tsk_Edata)
        if n < length(tStart_aEpoch)
            tsk_Edata=  Threads.@spawn getEdata(tStart_aEpoch[n+1])
        end
        dataGPU=(X_Einit=data.X_Einit,X_Efore=data.X_Efore) |>gpu
        predE=modelE(dataGPU.X_Einit,dataGPU.X_Efore) |>cpu        
        # --------------------- for RX -------------------- 
        Y_isValid=data.Y_isValid[:,:,:,1:24]
        RXmesh=fetch(tsk_Rdata)
        if n < length(tStart_aEpoch)
            tsk_Rdata=  Threads.@spawn getRX(tStart_aEpoch[n+1])
        end       
        X_allStart[n]=getValidSample(RXmesh,Y_isValid)
        Yfore_allStart=getValidSample(data.Y_fore[:,:,:,1:24],Y_isValid)
        PredE_allStart=getValidSample( predE[:,:,:,1:24],Y_isValid)
        Y_allStart[n]=vcat(Yfore_allStart-PredE_allStart,Yfore_allStart)
        if progressBar; ProgressMeter.next!(p;showvalues=[("loss",loss)]); end
    end
    trainX=hcat(X_allStart...)
    trainY=hcat(Y_allStart...)    
    # screen samples ----------------
    ratioLimit=7
    q_valid= vec(abs.(trainY[1,:]) .< ratioLimit * std(trainY[1,:])) 
    trainX=trainX[:,q_valid];trainY=trainY[:,q_valid];
    return trainX,trainY
end

# constructing dataLoader ----------------
tStart_allTrain=vcat(
    ZonedDateTime.(DateTime(2022,4,1,12):Day(1):(DateTime(2022,5,1,12)-Day(1)),tz"UTC"),
    ZonedDateTime.(DateTime(2022,7,1,12):Day(1):(DateTime(2022,8,1,12)-Day(1)),tz"UTC"),
    ZonedDateTime.(DateTime(2022,10,1,12):Day(1):(DateTime(2022,11,1,12)-Day(1)),tz"UTC")
);
dataE_info=(xVar_Einit=xVar_Einit,xVar_Efore=xVar_Efore,yVarName=yVarName,
    leads2init=leads2init,leads2Fore=leads2Fore,pInfo=pInfo,normLayer=normLayer
    )
tStart_all=ZonedDateTime.(DateTime(2023,1,1,12):Day(20):DateTime(2023,12,31,12),tz"UTC");
testE_looper=reset!(looperS(tStart_all,dataE_info,false));

data=next(testE_looper)
X_Einit=data.X_Einit |>gpu;
X_Efore=data.X_Efore |>gpu;
Y_isValid=data.Y_isValid  |>gpu;
Y_fore=data.Y_fore |>gpu;
getGradient_modelE(modelE,X_Einit,X_Efore,Y_fore,Y_fore,Y_isValid,sum(Y_isValid));
function mainTrain(tStart_allTrain,modelE,modelR,dataE_info,mInfo,testE_looper;progressBar=true,logFile=logFile,outputFolder=outputFolder)
    opt_E = Adam(0.002,(0.99, 0.999))
    opt_R = Adam(0.0001,(0.99, 0.999))
    opt_stateE = Flux.setup(opt_E, modelE) |> gpu;
    opt_stateR = Flux.setup(opt_R, modelR) |> gpu;    
    # start training ------------
    numStart_aBatch=36
    nRChunk_aEpoch=36;
    for epoch in 1:1000
        tStart_aEpoch=rand(tStart_allTrain,numStart_aBatch)
        isTrainModelR=0 < epoch  ? true : false
        isCalModelR = 0 < epoch  ? true : false
        GC.gc();CUDA.reclaim()
        # train the model ====================================
        modelE,opt_stateE,ELoss=trainModelE!(modelE,modelR,opt_stateE,dataE_info,tStart_aEpoch;epoch=epoch,calModelR=isCalModelR,mInfo=mInfo)
        if isTrainModelR
            modelR,opt_stateR,RLoss=trainModelR!(modelE,modelR,opt_stateR,dataE_info,mInfo,tStart_aEpoch;progressBar=progressBar,nRChunk_aEpoch=36)
        end
        # test the model ===================================
        if  rem(epoch,1)==0
            Flux.testmode!(modelE);Flux.testmode!(modelR);
            testLoss=getTestLoss(modelE,modelR,testE_looper;progressBar=true,mInfo=mInfo,calModelR=isCalModelR)
            outString="Epoch $(epoch): loss for train: $(mean(ELoss))     loss for test: $(testLoss[1]) $(testLoss[2]) $(testLoss[3]) "
            println(outString)
            # write to log file
            open(logFile,"a") do io
                println(io,outString)
            end
        end
        # save model
        model_stateE = Flux.state(modelE|>cpu);
        model_stateR = Flux.state(modelR|>cpu);
        opt_stateE = opt_stateE |>cpu
        opt_stateR = opt_stateR |>cpu
        jldsave("$outputFolder/model_epoch$(epoch).jld2"; model_stateE,opt_E,opt_stateE,model_stateR,opt_R,opt_stateR,numStart_aBatch,mInfo)

        opt_stateE = opt_stateE |>gpu 
        opt_stateR = opt_stateR |>gpu 
    end
    return modelE,modelR,opt_stateE,opt_stateR
end
modelE,modelR,opt_stateE,opt_stateR=mainTrain(tStart_allTrain,modelE,modelR,dataE_info,mInfo,testE_looper;progressBar=true,logFile=logFile,outputFolder=outputFolder)

