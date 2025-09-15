using ProNetM,Dates,DataFrames,Statistics,TimeZones
pInfo=ProNetM.comData.pInfo


# data processing ==========================
function addADim(dataIn)
    return reshape(dataIn,size(dataIn)...,1)
end
function getData_aStart(tStart,xVar_Einit,xVar_Efore,yVarName,leads2init,leads2Fore;pInfo=ProNetM.comData.pInfo,tLastLimit=nothing)
    # function getData_aStart(tStart,xVar_Einit,xVar_Efore,xVar_Dfore,oVar_Dbase,yVarName,leads2init,leads2Fore;pInfo=getProjectInfo())
    # println(tStart,xNaqVar_all,yVarName,epoch_all,leads_all)
    pollName=replace(yVarName, "erd_" => "", "_Err" => "", "_Ana" => "")
    tStart_file=astimezone(tStart,tz"UTC")
    Times2init=tStart_file .+ leads2init
    Times2Fore=tStart_file .+ leads2Fore

    tsk_Y_fore= Threads.@spawn addADim(readAnalysis_aStart(tStart_file,yVarName,Times2Fore;grdre=copy(pInfo.grdRe),datFileFun=pInfo.reFileFun))
    tsk_Y_isValid= Threads.@spawn addADim(readValidGrid(tStart_file,pollName,Times2Fore;grdre=copy(pInfo.grdRe),datFileFun=pInfo.reFileFun,distThreadhould=pInfo.validLength))
    tsk_xData_Einit=  Threads.@spawn addADim(readInitBeforeStart(Times2init, xVar_Einit; startLead=4, disp=false, grdIn=pInfo.grdNaq, datFileFun=pInfo.foreFileFun))
    tsk_xData_Efore= Threads.@spawn addADim(readFore_aStart(tStart_file,xVar_Efore,Times2Fore;grdIn=copy(pInfo.grdNaq), datFileFun=pInfo.foreFileFun))
    # set data exceed 20230101 to invalid
    if isnothing(tLastLimit )
        Y_isValid=fetch(tsk_Y_isValid)
    else
        qt_isvalid=reshape(Times2Fore .< tLastLimit,1,1,1,:,1)
        Y_isValid=fetch(tsk_Y_isValid) .& qt_isvalid
    end
    return (Y_fore=fetch(tsk_Y_fore),Y_isValid=Y_isValid,xData_Einit=fetch(tsk_xData_Einit),  xData_Efore=fetch(tsk_xData_Efore))
end


function dataProcessor(data,info)
    tsk_X_Einit= Threads.@spawn info.normLayer(data.xData_Einit[1:end-1,:,:,:,:],info.xVar_Einit)
    tsk_X_Efore= Threads.@spawn info.normLayer(data.xData_Efore[1:end-1,:,:,:,:],info.xVar_Efore)
    
    tsk_Y_isValid= Threads.@spawn data.Y_isValid[1:end-1,:,:,:,:]
    tsk_Y_fore= Threads.@spawn data.Y_fore[1:end-1,:,:,:,:]
    dataOut=(
        X_Einit=fetch(tsk_X_Einit),
        X_Efore=fetch(tsk_X_Efore),
        Y_isValid=fetch(tsk_Y_isValid),
        Y_fore=fetch(tsk_Y_fore)
    )
    return dataOut
end

mutable struct looperS
    tStart_all::AbstractArray
    idx_all::AbstractArray
    nextIdx::Int
    shuffle::Bool
    data_info::NamedTuple
    tReaded::ZonedDateTime
end
function looperS(tStart_all,data_info,isShuffle)
    return reset!(looperS(tStart_all,[],1,isShuffle,data_info,ZonedDateTime(2000,tz"UTC")))
end
function reset!(l::looperS)
    if l.shuffle
        l.idx_all=collect(eachobs(1:length(l.tStart_all),shuffle=true))
    else
        l.idx_all=1:length(l.tStart_all)
    end
    l.nextIdx=1
    return l
end
 
function next(l::looperS)
    tStart=l.tStart_all[l.idx_all[l.nextIdx]]
    info=l.data_info
    data=dataProcessor(getData_aStart(tStart,info.xVar_Einit,info.xVar_Efore,info.yVarName,info.leads2init,info.leads2Fore;pInfo=info.pInfo),info)
    if l.nextIdx >= length(l.tStart_all)
        reset!(l)
    else
        l.nextIdx +=1
    end
    l.tReaded=tStart
    return data
end


# define model ==========================
mutable struct EmissionModelCell
    encSpaceInit::Chain
    encSpaceFore::Chain
    init_foreMapper1::Chain
    init_foreMapper2::Chain
    init_foreMapper3::Chain
    hidMap_inToHid::Chain
    hidMap_foreToHid::Chain
    hid::Chain
    decSpace::Chain
end
function (m::EmissionModelCell)(x_STinit,x_STfore) 
    W_init,H_init,C_init,T_init,B= size(x_STinit)
    W_fore,H_fore,C_fore,T_fore,B= size(x_STfore)
    # encoder ------------------    
    # encode init
    # println(size(x_STinit))
    x_STinit,inT,B=ProNetM.toSpaceForm(x_STinit)
    x_STinit=m.encSpaceInit[1:1](x_STinit)
    # println(size(x_STinit))
    skip_init_1,H_1,C_space1=ProNetM.spaceToTimeForm(x_STinit,T_init,B)
    x_STinit=m.encSpaceInit[2:4](x_STinit)    
    # println(size(x_STinit))
    skip_init_2,H_2,C_space2=ProNetM.spaceToTimeForm(x_STinit,T_init,B)
    x_STinit=m.encSpaceInit[5:end](x_STinit)    
    # println(size(x_STinit))
    skip_init_3,H_3,C_space3=ProNetM.spaceToTimeForm(x_STinit,T_init,B)
    x_STinit,_,_=ProNetM.spaceToHidForm(x_STinit,T_init,B)
    x_STinit=m.hidMap_inToHid(x_STinit)    
    # println(size(x_STinit))
    # encode fore
    x_STfore,outT,B=ProNetM.toSpaceForm(x_STfore)
    x_STfore=m.encSpaceFore[1:1](x_STfore)
    skip_fore_1=copy(x_STfore)
    x_STfore=m.encSpaceFore[2:4](x_STfore)
    skip_fore_2=copy(x_STfore)
    x_STfore=m.encSpaceFore[5:end](x_STfore)
    skip_fore_3=copy(x_STfore)
    x_STfore,_,_=ProNetM.spaceToHidForm(x_STfore,T_fore,B)
    x_STfore=m.hidMap_foreToHid(x_STfore)
    
    # hidden ------------------
    x=cat(x_STinit,x_STfore;dims=3)
    x=m.hid(x)
    # x=backFromHiddenForm(x,C,T)
    x,_,_=ProNetM.hidToSpaceForm(x,C_space3,T_fore)

    # bridge between inT and outT ------------------
    skip_init_3=m.init_foreMapper3(skip_init_3)
    skip_init_3,_,_=ProNetM.timeToSpaceForm(skip_init_3,H_3,C_space3)
    skip_init_2=m.init_foreMapper2(skip_init_2)
    skip_init_2,_,_=ProNetM.timeToSpaceForm(skip_init_2,H_2,C_space2)
    skip_init_1=m.init_foreMapper1(skip_init_1)
    skip_init_1,_,_=ProNetM.timeToSpaceForm(skip_init_1,H_1,C_space1)

    # decoder ------------------ 
    x=cat(x+skip_fore_3,x+skip_init_3;dims=3)
    x=m.decSpace[1:4](x)
    x=cat(x+skip_fore_2,x+skip_init_2;dims=3)
    x=m.decSpace[5:7](x)
    x=cat(x+skip_fore_1,x+skip_init_1;dims=3)
    x=m.decSpace[8:end](x)
    x=ProNetM.backFromSpaceForm( x ,outT,B)
    return  x
end
Flux.@layer EmissionModelCell

function getEmissionModel(inT,outT,inC_Einit,inC_Efore,outC,hid_S,hid_T;nGroup_space=[1,8,16],nGroup_hid=8,drop_prob=0.2)
    # encoder decoder for Space -----------------------
    encSpaceInit =Chain(
        Conv((3, 3), inC_Einit => hid_S[1], swish;stride=1,pad = 1),
        Conv((3, 3), hid_S[1] => hid_S[2], swish;stride=2,pad = 1),
        # GroupNorm(hid_S[2],nGroup_space[2]),
        Conv((3, 3), hid_S[2] => hid_S[2], swish;stride=1,pad = 1),
        GroupNorm(hid_S[2],nGroup_space[2]),
        Conv((3, 3), hid_S[2] => hid_S[3], swish;stride=2,pad = 1),
        # GroupNorm(hid_S[3],nGroup_space[3]),
        Conv((3, 3), hid_S[3] => hid_S[3], swish;stride=1,pad = 1),
        GroupNorm(hid_S[3],nGroup_space[3]),
    )# |> device
    encSpaceFore =Chain(
        Conv((3, 3), inC_Efore => hid_S[1], swish;stride=1,pad = 1),
        Conv((3, 3), hid_S[1] => hid_S[2], swish;stride=2,pad = 1),
        # GroupNorm(hid_S[2],nGroup_space[2]),
        Conv((3, 3), hid_S[2] => hid_S[2], swish;stride=1,pad = 1),
        GroupNorm(hid_S[2],nGroup_space[2]),
        Conv((3, 3), hid_S[2] => hid_S[3], swish;stride=2,pad = 1),
        # GroupNorm(hid_S[3],nGroup_space[3]),
        Conv((3, 3), hid_S[3] => hid_S[3], swish;stride=1,pad = 1),
        GroupNorm(hid_S[3],nGroup_space[3]),
    ) 

    init_foreMapper1=Chain(
        Dropout(drop_prob), 
        Conv((1, 1), inT => outT, swish) 
    )
    init_foreMapper2=Chain(
        Dropout(drop_prob), 
        Conv((1, 1), inT => outT, swish) 
    )
    init_foreMapper3=Chain(
        Dropout(drop_prob), 
        Conv((1, 1), inT => outT, swish) 
    )

    decSpace = Chain(
        GroupNorm(2*hid_S[3],nGroup_space[3]),
        Conv((3, 3), 2*hid_S[3] => hid_S[2]*4, swish;stride=1,pad = 1),
        PixelShuffle(2),
        # GroupNorm(hid_S,nGroup_space),
        Conv((3, 3), hid_S[2] => hid_S[2], swish;stride=1,pad = (1,1)),
        GroupNorm(2*hid_S[2],nGroup_space[2]),
        Conv((3, 3), 2*hid_S[2] => hid_S[1]*4, swish;stride=1,pad = 1),
        PixelShuffle(2),
        # Conv((3, 3), hid_S[1] => hid_S[1], swish;stride=1,pad = 1),
        Conv((3, 3), 2*hid_S[1] => hid_S[1], swish;stride=1,pad = 1),
        Conv((3, 3), hid_S[1] => outC ;stride=1,pad = 1)        
    ) 
    # hidden -----------------------
    hidMap_inToHid=Chain(
        Dropout(drop_prob),    
        Conv((1, 1), inT*hid_S[3] => hid_T)# |> device
    )
    hidMap_foreToHid=Chain(
        Dropout(drop_prob),    
        Conv((1, 1), outT*hid_S[3] => hid_T)# |> device
    )
    hid = Chain(
        ProNetM.GASubBlock(2*hid_T; kernel_size=21, drop_prob=drop_prob,nGroup=nGroup_hid),
        ProNetM.Conv((1, 1), 2*hid_T => hid_T), # reduce channel to hid_T
        ProNetM.GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/2,nGroup=nGroup_hid),
        ProNetM.GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/4,nGroup=nGroup_hid),
        ProNetM.GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/6,nGroup=nGroup_hid),
        # GASubBlock(hid_T; kernel_size=21, drop_prob=drop_prob/8,nGroup=nGroup_hid),
        # GASubBlock(hid_T; kernel_size=21, drop_prob=0.0,nGroup=nGroup_hid),
        Dropout(drop_prob),    
        Conv((1, 1), hid_T=>outT*hid_S[3] ), # reduce channel to hid_T
    )
    modelST=EmissionModelCell(encSpaceInit,encSpaceFore,init_foreMapper1,init_foreMapper2,init_foreMapper3,hidMap_inToHid,hidMap_foreToHid,hid,decSpace) 
    return modelST
end


# ================================= model Initialize ====================================
function getModelST(;
    xVar_Einit=["U","V","windSpeed","windDiv","Ter","Land","holiday","TEMP","RH"],
    xVar_Efore=["U","V","windSpeed","windDiv","Ter","Land","holiday","TEMP","RH"],
    # xVar_Dfore=["U","V","windSpeed","windDiv","Ter","Land","TEMP","RH"],
    # oVar_Dbase=["PM25","PM25_Err"],
    yVarName="PM25",
    initNorm=true,
    hid_S=[8,32,128],hid_T=512,#hid_D=128,
    nGroup_space=[1,8,16],nGroup_hid=32, drop_prob=0.2,
    inT=176,outT=176,
    leads2init=Hour.(-(inT-1):0),
    leads2Fore=Hour.(1:outT),
    inC_Einit=length(xVar_Einit),
    inC_Efore=length(xVar_Efore),
    # inC_Dfore=length(xVar_Dfore),
    outC=1
    )

    # constructing the model -------------------------
    modelE=getEmissionModel(inT,outT,inC_Einit,inC_Efore,outC,hid_S,hid_T;nGroup_space=nGroup_space,nGroup_hid=nGroup_hid,drop_prob=drop_prob)

    # constructing normLayer ----------------
    var_all=unique(vcat(xVar_Einit,xVar_Efore,[yVarName]))
    # var_all=unique(vcat(xVar_Einit,xVar_Efore,xVar_Dfore))
    q_zeroMean=[startswith(x,"erd_") || endswith(x,"_Err") for x in var_all]
    Times2init=ZonedDateTime.(vcat(collect(DateTime(2022) .+ Day(d) .+ Hour.(0:23) for d in 0:15:365)...),tz"Asia/Shanghai")
    if initNorm
        X_XTinit=readInitBeforeStart(Times2init,var_all;startLead=4)
        normLayer=NormPoll(X_XTinit,var_all;q_zeroMean=q_zeroMean) 
    else
        normLayer=nothing;
    end
        
    info=(xVar_Einit=xVar_Einit,
        xVar_Efore=xVar_Efore,
        # oVar_Dbase=oVar_Dbase,
        yVarName=yVarName,
        inT=inT, outT=outT,
        inC_Einit=inC_Einit,inC_Efore=inC_Efore,
        hid_S=hid_S, hid_T=hid_T,# hid_D=hid_D,
        nGroup_space=nGroup_space,nGroup_hid=nGroup_hid,drop_prob=drop_prob,
        outC=outC,
        leads2init=leads2init,
        leads2Fore=leads2Fore,
        normLayer=normLayer)
        # return modelE,modelD,normLayer,info
    return modelE,normLayer,info
end
