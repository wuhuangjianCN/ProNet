using ProNetM,Dates,DataFrames,Statistics,TimeZones

pInfo=ProNetM.comData.pInfo
function readEpoches_aStart(tStart::ZonedDateTime,varName::AbstractString,epoch_all::AbstractVector,Times2read,datFileFun;grdIn=copy(comData.pInfo.grdNaq),varInCtl="data")

    dataOut=Array{Float32,4}(undef,grdIn.lonlatsize...,length(epoch_all),length(Times2read))
    Threads.@threads for i in eachindex(epoch_all)
        epoch=epoch_all[i]
        ProNetM.readFore_aStart!(view(dataOut,:,:,i:i,:),tStart,varName,Times2read;grdIn=copy(grdIn),datFileFun=(tStart,varName) -> datFileFun(tStart,varName,epoch),varInCtl=varInCtl)
    end
    # end
    return dataOut
end

function addADim(dataIn)
    return reshape(dataIn,size(dataIn)...,1)
end
function getData_aStart(tStart,xNaqVar_all,yVarName,epoch_all,leads_all;pInfo=ProNetM.comData.pInfo,expTag="EDshift_V101",tLastLimit=nothing)
    tStart_file=astimezone(tStart,tz"UTC")
    Times_aStart=tStart_file .+ leads_all
    dataFolder=pInfo.dataFolder
    datFun_ProNetM(tStart,varName,epoch)="$(dataFolder)/outputs/$expTag/epoch$epoch/$varName.$(Dates.format(tStart,"yyyymmddHH")).dat"
    datFun_Naq(tStart,varName)="$(dataFolder)/Naqpms/$(Dates.format(tStart,"yyyymmddHH"))/$varName.$(Dates.format(tStart,"yyyymmddHH")).dat"
    datFun_Re(tStart,varName)="$(dataFolder)/reanalysis/$varName/$varName.$(Dates.format(tStart,"yyyymmddHH")).grd"

    tsk_epoches= Threads.@spawn readEpoches_aStart(tStart_file,yVarName,epoch_all,Times_aStart,datFun_ProNetM;grdIn=copy(pInfo.grdNaq))
    naq_aStart=readFore_aStart(tStart_file,xNaqVar_all,Times_aStart;grdIn=copy(pInfo.grdNaq), datFileFun=datFun_Naq)
    Y_fore=readAnalysis_aStart(tStart_file,yVarName,Times_aStart;grdre=copy(pInfo.grdRe),datFileFun=datFun_Re)
    Y_isValid=readValidGrid(tStart_file,yVarName,Times_aStart;grdre=copy(pInfo.grdRe),datFileFun=datFun_Re,distThreadhould=pInfo.validLength,)

    if !isnothing(tLastLimit )
        qt_isvalid=reshape(Times_aStart .< tLastLimit,1,1,1,:)
        Y_isValid=Y_isValid .& qt_isvalid
    end

    epoches_aStart=fetch(tsk_epoches)
    return (X_naq=addADim(naq_aStart),X_ens=addADim(epoches_aStart),Y_fore=addADim(Y_fore),Y_isValid=addADim(Y_isValid))
end


# define model ==========================
mutable struct EnsembleModelCell
    encSpace::Chain
    hid::Chain
    decRatio::Chain
    decCon::Chain
    tMap_Ratio::Chain
    tMap_Con::Chain
end
function (m::EnsembleModelCell)(X,ensemble)    
    W_fore,H_fore,C_in,T_in,B= size(X)
    # encoder ------------------    
    X,inT,B=ProNetM.toSpaceForm(X)
    X=m.encSpace[1](X)
    skip_fore_1=copy(X)
    X=m.encSpace[2:4](X)
    skip_fore_2=copy(X)
    X=m.encSpace[5:end](X)
    skip_fore_3=copy(X)
    # hidden ------------------
    x,C_hid,T_hid=ProNetM.spaceToHidForm(X,T_in,B)
    x=m.hid(x)
    xRatio=m.tMap_Ratio(x)
    xCon=m.tMap_Con(x)
    xRatio,_,_=ProNetM.hidToSpaceForm(xRatio,C_hid,T_hid)
    xCon,_,_=ProNetM.hidToSpaceForm(xCon,C_hid,T_hid)
    # decoder ------------------ 
    xRatio=xRatio + skip_fore_3
    xRatio=m.decRatio[1:4](xRatio)
    xRatio=xRatio+skip_fore_2
    xRatio=m.decRatio[5:7](xRatio)
    xRatio=xRatio+skip_fore_1
    xRatio=m.decRatio[8:end](xRatio)
    xRatio=ProNetM.backFromSpaceForm( xRatio ,inT,B)
    xRatio=softmax(xRatio;dims=3)
    # xCon
    # xCon=m.decCon[1:1](xCon)
    xCon=xCon + skip_fore_3
    xCon=m.decCon[1:4](xCon)
    xCon=xCon+skip_fore_2
    xCon=m.decCon[5:7](xCon)
    xCon=xCon+skip_fore_1
    xCon=m.decCon[8:end](xCon)
    xCon=ProNetM.backFromSpaceForm( xCon ,inT,B)
   
    return  sum(ensemble .* xRatio;dims=3) .+ xCon
end
Flux.@layer EnsembleModelCell

function getEnsembleModel(inC,inT,hid_S,hid_T,outC_ratio;nGroup_space=[4,8],nGroup_hid=8,drop_prob=0.2)
    # encoder decoder for Space -----------------------
    encSpaceFore =Chain(
        Conv((3, 3),  inC => hid_S[1], swish;stride=1,pad = 1),
        Conv((3, 3), hid_S[1] => hid_S[2], swish;stride=2,pad = 1),
        Conv((3, 3), hid_S[2] => hid_S[2], swish;stride=1,pad = 1),
        GroupNorm(hid_S[2],nGroup_space[2]),
        Conv((3, 3), hid_S[2] => hid_S[3], swish;stride=2,pad = 1),
        Conv((3, 3), hid_S[3] => hid_S[3], swish;stride=1,pad = 1),
        GroupNorm(hid_S[3],nGroup_space[3])
    ) 
    decRatio = Chain(
        GroupNorm(hid_S[3],nGroup_space[3]),
        Conv((3, 3), hid_S[3] => hid_S[2]*4, swish;stride=1,pad = 1),
        PixelShuffle(2),
        Conv((3, 3), hid_S[2] => hid_S[2], swish;stride=1,pad = (1,1)),
        GroupNorm(hid_S[2],nGroup_space[2]),
        Conv((3, 3),  hid_S[2] =>  hid_S[1]*4, swish;stride=1,pad = 1),
        PixelShuffle(2),
        Conv((3, 3), hid_S[1] => hid_S[1], swish;stride=1,pad = 1),
        Conv((3, 3), hid_S[1] => outC_ratio ;stride=1,pad = 1)
    ) 
    decCon = Chain(
        GroupNorm(hid_S[3],nGroup_space[3]),
        Conv((3, 3), hid_S[3] => hid_S[2]*4, swish;stride=1,pad = 1),
        PixelShuffle(2),
        Conv((3, 3), hid_S[2] => hid_S[2], swish;stride=1,pad = (1,1)),
        GroupNorm(hid_S[2],nGroup_space[2]),
        Conv((3, 3),  hid_S[2] =>  hid_S[1]*4, swish;stride=1,pad = 1),
        PixelShuffle(2),
        Conv((3, 3), hid_S[1] => hid_S[1], swish;stride=1,pad = 1),
        Conv((3, 3), hid_S[1] => 1 ;stride=1,pad = 1)
    ) 
    # hidden -----------------------
    hid = Chain(
        Conv((1, 1), inT*hid_S[3] => hid_T), # reduce channel to hid_T
        ProNetM.GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob,nGroup=nGroup_hid),
        ProNetM.GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/2,nGroup=nGroup_hid),
        ProNetM.GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/4,nGroup=nGroup_hid),
    )
    tMap_Ratio=Chain(    Conv((1, 1), hid_T=>inT*hid_S[3] )  )# increase channel to outputs   
    tMap_Con=Chain(    Conv((1, 1), hid_T=>inT*hid_S[3] )  )# increase channel to outputs   
    modelEnsemble=EnsembleModelCell(encSpaceFore,hid,decRatio,decCon,tMap_Ratio,tMap_Con) 
    return modelEnsemble
end

    function dataProcessor(data,normLayer)
        tsk_X_naq= Threads.@spawn normLayer(data.X_naq[1:end-1,:,:,:,:],normLayer.varName)
        tsk_X_ens= Threads.@spawn data.X_ens[1:end-1,:,:,:,:]
        tsk_Y_isValid= Threads.@spawn data.Y_isValid[1:end-1,:,:,:,:]
        tsk_Y_fore= Threads.@spawn data.Y_fore[1:end-1,:,:,:,:]
        pollName=normLayer.varName[1]
        pollMean,pollStd=ProNetM.getMeanStdV(normLayer,[pollName])
        X_naq=fetch(tsk_X_naq)
        ensemble=fetch(tsk_X_ens)
        tsk_X= Threads.@spawn cat(X_naq,(ensemble .- pollMean)./ pollStd ;dims=3)
        dataOut=(
            X=fetch(tsk_X),
            ensemble=ensemble,
            Y_isValid=fetch(tsk_Y_isValid),
            Y_fore=fetch(tsk_Y_fore)
        )
        return dataOut
    end

function initModel_Ensemble(pollName;
    xNaqVar_all=[pollName,"erd_$pollName","U","V","windSpeed","windDiv","Ter","Land","holiday","TEMP","RH"],
    epoch_all=91:100,
    yVarName=pollName,
    leads_all=Hour.(1:inT),
    initNorm=true,
    hid_S=[8,16,128],hid_T=512,
    nGroup_space=[1,8,16],nGroup_hid=8,    drop_prob=0.2,
    inT=length(leads_all),
    inC=length(xNaqVar_all) + length(epoch_all),
    outC_ratio=length(epoch_all)
    )

    # constructing the model -------------------------
    modelEnsemble=getEnsembleModel(inC,inT,hid_S,hid_T,outC_ratio;nGroup_space=nGroup_space,nGroup_hid=nGroup_hid,drop_prob=drop_prob)

    # constructing normLayer ----------------
    Times2norm=ZonedDateTime.(vcat(collect(DateTime(2022) .+ Day(d) .+ Hour.(0:23) for d in 0:15:365)...),tz"Asia/Shanghai")
    if initNorm
        X2norm=readInitBeforeStart(Times2norm,xNaqVar_all;startLead=4)
        normLayer=NormPoll(X2norm,xNaqVar_all;dim_poll=3) 
    else
        normLayer=nothing;
    end
    info=(xNaqVar_all=xNaqVar_all,  
        epoch_all=epoch_all,  
        yVarName=yVarName,
        leads_all=leads_all,
        hid_S=hid_S,
        hid_T=hid_T,
        nGroup_space=nGroup_space,
        nGroup_hid=nGroup_hid,
        drop_prob=drop_prob,
        normLayer=normLayer
    )
        
    return modelEnsemble,normLayer,info
end
