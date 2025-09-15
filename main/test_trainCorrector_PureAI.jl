using CUDA,Random
Random.seed!(1234)
using ProNetM,CUDA,Dates,DataFrames,Statistics,TimeZones,JLD2,Flux,MLUtils,ProgressMeter
pInfo=getProjectInfo(dirname(pwd()));
pInfo=Base.setindex(pInfo,30000,:validLength);
ProNetM.comData=getCommenData(pInfo)
include("$(pInfo.baseFolder)/main/modelInit_PureAI.jl");
lonM,latM=getLonLatM();
expName="PureAI";
outputFolder="$(pInfo.dataFolder)/outputs/$(expName)";
if !isdir(outputFolder);    mkdir(outputFolder);  end
logFile="$outputFolder/log.$expName"

progressBar=true
# ============================ get the model ==========================
poll="PM25";
xVar_Einit=["U","V","windSpeed","windDiv","Ter","Land","TEMP","RH","PBLH"];
xVar_Efore=["U","V","windSpeed","windDiv","Ter","Land","TEMP","RH","PBLH"];
yVarName=poll;

inT=24*7;outT=24*7;
leads2init= Hour(3) .+ Hour.(-(inT-1):0);
leads2Fore=Hour(3) .+ Hour.(1:outT);

modelE,normLayer,mInfo=getModelST(;xVar_Einit=xVar_Einit,xVar_Efore=xVar_Efore,
    yVarName=yVarName,
    inT=inT,outT=outT,
    leads2init=leads2init,leads2Fore=leads2Fore,    
    hid_S=[8,16,128],hid_T=256,
    nGroup_space=[1,8,16],nGroup_hid=32,
    initNorm=true);
    
modelE=modelE |> gpu ;
function getTestLoss(modelE,test_looper;progressBar=false)  
    nBatch=length(test_looper.idx_all)
    loss_aEpoch=fill(NaN32,nBatch)
    num_aEpoch=fill(0,nBatch)
    reset!(test_looper)
    tsk_getData= Threads.@spawn next(test_looper);
    if progressBar; p = Progress(1+length(loss_aEpoch); desc="Testing:",dt=0.1); end
    for n in eachindex(loss_aEpoch)
        data=fetch(tsk_getData)|>gpu
        tsk_getData= Threads.@spawn next(test_looper);
        count=sum(data.Y_isValid)
        predE=modelE(data.X_Einit,data.X_Efore)
        loss=sum((predE - data.Y_fore).^2  .* data.Y_isValid)/count
        loss=sqrt(loss)
        loss_aEpoch[n]=loss|>cpu
        num_aEpoch[n]=count|>cpu
        if progressBar; ProgressMeter.next!(p;showvalues=[("loss",loss)]); end
    end
    GC.gc();CUDA.reclaim()
    q_valid=num_aEpoch.>0
    weight_all=num_aEpoch ./ sum(num_aEpoch)
    loss=sum(loss_aEpoch[q_valid].*weight_all[q_valid])
    if progressBar; ProgressMeter.next!(p; showvalues=[("loss for Test:",loss)]); end
    return loss
end
# ==================================== training ===============================
function getGradient_modelE(modelE,X_Einit,X_Efore,targetY_curr,Y_isValid,count)
    loss, gradsE = Flux.withgradient(modelE) do m
        predE=m(X_Einit,X_Efore)
        loss=sum((abs.(predE  - targetY_curr) ) .* Y_isValid)/count
    end
    return loss,gradsE
end
function trainModelE!(modelE,opt_stateE,eInfo,tStart_aEpoch;epoch=0,mInfo=mInfo)
    Flux.trainmode!(modelE);
    loss_aEpoch=fill(NaN32,length(tStart_aEpoch))
    num_aEpoch=fill(0,length(tStart_aEpoch))    
    
    tLastLimit=ZonedDateTime(2023,1,1,tz"Asia/Shanghai")
    getEdata(tStart)=dataProcessor(getData_aStart(tStart,eInfo.xVar_Einit,eInfo.xVar_Efore,eInfo.yVarName,eInfo.leads2init,eInfo.leads2Fore;pInfo=eInfo.pInfo,tLastLimit=tLastLimit),eInfo)
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
        targetY_curr=data.Y_fore |>gpu
        count=sum(Y_isValid)
        loss, gradsE = getGradient_modelE(modelE,X_Einit,X_Efore,targetY_curr,Y_isValid,count)
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

# constructing dataLoader ----------------
tStart_allTrain=ZonedDateTime.(DateTime(2021,12,25,12):Day(1):DateTime(2022,12,30,12),tz"UTC");
data_info=(xVar_Einit=xVar_Einit,xVar_Efore=xVar_Efore,yVarName=yVarName,
    leads2init=leads2init,leads2Fore=leads2Fore,pInfo=pInfo,normLayer=normLayer);
tStart_all=ZonedDateTime.(DateTime(2023,1,1,12):Day(20):DateTime(2023,12,31,12),tz"UTC");
test_looper=reset!(looperS(tStart_all,data_info,false));

# ==================================== training ===============================
data=dataProcessor(getData_aStart(tStart_allTrain[1],xVar_Einit,xVar_Efore,yVarName,leads2init,leads2Fore;pInfo=pInfo),data_info)
data=data|>gpu
count=sum(data.Y_isValid)
loss, gradsE = getGradient_modelE(modelE,data.X_Einit,data.X_Efore,data.Y_fore,data.Y_isValid,count)
data,gradsE=nothing,nothing
function mainTrain(tStart_allTrain,modelE,data_info,mInfo,test_looper;progressBar=true,logFile=logFile,outputFolder=outputFolder)
    opt_E =Adam(0.001,(0.99, 0.999)) ;
    opt_stateE = Flux.setup(opt_E, modelE) |> gpu;
    numStart_aBatch=36;
    for epoch in 1:700
        # epoch = 1
        tStart_aEpoch=rand(tStart_allTrain,numStart_aBatch)
        GC.gc();CUDA.reclaim()
        # train the model ====================================
        modelE,opt_stateE,ELoss=trainModelE!(modelE,opt_stateE,data_info,tStart_aEpoch;epoch=epoch,mInfo=mInfo)
        # test the model ===================================
        if  rem(epoch,1)==0
            Flux.testmode!(modelE);
            testLoss=getTestLoss(modelE,test_looper;progressBar=true)
            outString="Epoch $(epoch): loss for train: $(mean(ELoss))     loss for test: $(testLoss) "
            println(outString)
            # write to log file
            open(logFile,"a") do io
                println(io,outString)
            end
        end
        # save model
        model_stateE = Flux.state(modelE|>cpu);
        opt_stateE = opt_stateE |>cpu
        jldsave("$outputFolder/model_epoch$(epoch).jld2"; model_stateE,opt_E,opt_stateE,numStart_aBatch,mInfo)

        opt_stateE = opt_stateE |>gpu 
    end
    return modelE,opt_stateE
end
modelE,opt_stateE=mainTrain(tStart_allTrain,modelE,data_info,mInfo,test_looper;progressBar=true,logFile=logFile,outputFolder=outputFolder)
