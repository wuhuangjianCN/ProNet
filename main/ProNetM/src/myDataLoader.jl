using Dates,TimeZones,GradDataM,MLUtils,Base,StaDataM,CSV,DataFrames,XLSX
export dataInfoCell,readInitBeforeStart,readFore_aStart,writeFore_aStart,readAnalysis,readValidGrid,readMiniDist
export asyncLoaderCell,next!,next,getProjectInfo,getLonLatM,getStaObs,readAnalysis_aStart,writeAnalysis_aStart,getStaInfo
export CuLoader,comData,getCommenData
# dimensions of data are (lon,lat,var,time,batch)
# include("NormPoll.jl")
function getProjectInfo(baseFolder=dirname(dirname(pathof(@__MODULE__))))
    dataFolder="$baseFolder/DATA"
    validLength=80000
    DATAinfo=(
        validLength=validLength,
        baseFolder=baseFolder,
        dataFolder=dataFolder,
        grdGrid=gradData("$dataFolder/datagrid/grid.ctl"),
        grdNaq=gradData("$dataFolder/dataNaq.ctl"),
        obsFileFun=(tMonth)-> "$dataFolder/Obs/pollObsQC$(Dates.format(tMonth,"yyyymm")).mat",
        foreFileFun=(tStart,varName)->"$dataFolder/Naqpms/$(Dates.format(astimezone(tStart,tz"UTC"),"yyyymmddHH"))/$varName.$(Dates.format(astimezone(tStart,tz"UTC"),"yyyymmddHH")).dat",
        grdRe=gradData("$dataFolder/dataRe.ctl"),
        reFileFun=(tStart,varName)->"$dataFolder/reanalysis/$varName/$varName.$(Dates.format(astimezone(tStart,tz"UTC"),"yyyymmddHH")).grd",

        staPM25mid = "$dataFolder/总站人工预报/PM25_2023_mid.mat",
        staPM25obs = "$dataFolder/总站人工预报/PM25_2023_obs.mat",
        staO3mid = "$dataFolder/总站人工预报/O3_2023_mid.mat",
        staO3obs = "$dataFolder/总站人工预报/O3_2023_obs.mat",
    )
    return DATAinfo
end
function getLonLatM(;grdGrid=comData.pInfo.grdGrid)
    # grdGrid=gradData(fileName)
    lonM=read(grdGrid,"XLONG",1);
    latM=read(grdGrid,"XLAT",1);
    return lonM,latM
end
function getCommenData(pInfo=getProjectInfo())
    
    grdGrid=pInfo[:grdGrid]
    lonM,latM=getLonLatM(;grdGrid=grdGrid)
    ter=read(grdGrid,"Ter",1)
    land=read(grdGrid,"Land",1)
    
    # prepare holidays data
    df_holiday = DataFrame(XLSX.readtable("$(pInfo.dataFolder)/holidays.xlsx", "Sheet1"))
    df_holiday.date=ZonedDateTime.(DateTime.(df_holiday.date),tz"Asia/Shanghai")
    df_holiday.holidays=Float32.(df_holiday.holidays)
    return (lonM=lonM,latM=latM,ter=ter,land=land,pInfo=pInfo,df_holiday=df_holiday)
end
comData=getCommenData()
function getStaInfo(fileName="staInfo_all.csv"::AbstractString;dataFolder=comData.pInfo.dataFolder)
    fullFileName="$(dataFolder)/staInfo/$fileName"
    staInfo=CSV.read(fullFileName,DataFrame)
    return staInfo
end
function getStaObs(Times;obsFileFun=comData.pInfo.obsFileFun)
    if !(Times isa AbstractArray);Times=[Times];end
    # get the unique tMonth in Times
    tMonth_all=unique(floor.(Times,Month))
    files=obsFileFun.(tMonth_all)
    Times_all=minimum(tMonth_all):Hour(1):(maximum(tMonth_all)+Month(1)-Hour(1))
    staObs=staDATA(files;Times=Times_all)
    return staObs
end
# ================================== readInitBeforeStart =============================
function readInitBeforeStart(Times::AbstractVector{ZonedDateTime},varName_all::AbstractVector;grdIn=comData.pInfo.grdNaq,datFileFun=comData.pInfo.foreFileFun,startLead=0,disp=true)
    dataOut=Array{Float32,4}(undef,grdIn.lonlatsize...,length(varName_all),length(Times))
    Threads.@threads   for n in eachindex(varName_all)
        varName=varName_all[n]
        # println("Thread ID:",Threads.threadid())
        readInitBeforeStart!(view(dataOut,:,:,n:n,:),Times,varName;grdIn=copy(grdIn),datFileFun=datFileFun,startLead=startLead,disp=disp)
    end
    return dataOut
end

function readInitBeforeStart(Times::AbstractVector{ZonedDateTime},varName::AbstractString;grdIn=copy(comData.pInfo.grdNaq),datFileFun=comData.pInfo.foreFileFun,startLead=0,disp=true,varInCtl="data",comData=comData)
    
    dataOut=Array{Float32,4}(undef,grdIn.lonlatsize...,1,length(Times))
    # deal with unusual varName -------------------------------------
    if uppercase(varName)=="LAND"
        dataOut .= comData.land
        return dataOut
    elseif uppercase(varName)=="TER"
        dataOut .= comData.ter
        return dataOut
    elseif uppercase(varName)=="HOLIDAY"
        idx=indexin(floor.(astimezone.(Times,tz"Asia/Shanghai"),Day),comData.df_holiday.date)
        dataOut .= reshape(comData.df_holiday.holidays[idx],1,1,1,length(Times))
        return dataOut
    end
    # deal with usual varName -------------------------------------
    Times=astimezone.(Times,tz"UTC")    
    tStart_all=floor.(Times-Hour(12+startLead),Day)+Hour(12)
    # iterate by tStart or file, avoid open the same file multiple times
    for tStart in sort(unique(tStart_all))
        # find the leads in current tStart to read
        q_currStart= tStart .== tStart_all
        tRange=findfirst(q_currStart):findlast(q_currStart)
        Times_currStart=Times[tRange]
        leads_currStart=Dates.value.(Hour.(Times_currStart .- tStart))
        nZ_range=(minimum(leads_currStart):maximum(leads_currStart)) .+ 1
        grdIn.datFileName=datFileFun(tStart,varName)
        if isfile(grdIn.datFileName)
            read!(view(dataOut,:,:,:,tRange),grdIn,varInCtl,nZ_range)
        else
            @warn "File $(grdIn.datFileName) does not exist"
            return nothing
        end
    end
    return dataOut
end

function readInitBeforeStart!(dataReaded,Times::AbstractVector{ZonedDateTime},varName::AbstractString;grdIn=copy(comData.pInfo.grdNaq),datFileFun=comData.pInfo.foreFileFun,startLead=0,disp=true,varInCtl="data",comData=comData)

    # deal with unusual varName -------------------------------------
    if uppercase(varName)=="LAND"
        dataReaded .= comData.land
        return dataReaded
    elseif uppercase(varName)=="TER"
        dataReaded .= comData.ter
        return dataReaded
    elseif uppercase(varName)=="LEADTIME"
        leadTimes=Int.(round.(7*24 .* rand(length(Times))))
        @warn "Reading Random leadTime in readInitBeforeStart"
        dataReaded .= reshape(leadTimes,1,1,1,:)
        return dataReaded
    elseif uppercase(varName)=="HOLIDAY"
        idx=indexin(floor.(astimezone.(Times,tz"Asia/Shanghai"),Day),comData.df_holiday.date)
        # println(Times)
        dataReaded .= reshape(comData.df_holiday.holidays[idx],1,1,1,length(Times))
        return dataReaded
    end

    # deal with usual varName -------------------------------------
    Times=astimezone.(Times,tz"UTC")
    tStart_all=floor.(Times-Hour(12+startLead),Day)+Hour(12)
    
    dataBuffer=Array{Float32,3}(undef,grdIn.lonlatsize...,24)
    # iterate by tStart or file, avoid open the same file multiple times
    # Threads.@threads  
    for tStart in sort(unique(tStart_all))
        # find the leads in current tStart to read
        # grd_curr=copy(grdIn)
        grd_curr=grdIn
        q_currStart= tStart .== tStart_all
        tRange=findfirst(q_currStart):findlast(q_currStart)
        Times_currStart=Times[tRange]
        leads_currStart=Dates.value.(Hour.(Times_currStart .- tStart))
        nZ_range=(minimum(leads_currStart):maximum(leads_currStart)) .+ 1

        grd_curr.datFileName=datFileFun(tStart,varName)
        buffer_curr=view(dataBuffer,:,:,1:length(nZ_range))
        if isfile(grd_curr.datFileName)
            read!(buffer_curr,grd_curr,varInCtl,nZ_range)
            dataReaded[:,:,:,tRange]=buffer_curr
        else
            @warn "File $(grd_curr.datFileName) does not exist"
            return nothing
        end
    end
    dataBuffer=nothing
    return dataReaded
end

# ================================== readFore_aStart =============================
function readFore_aStart(tStart::ZonedDateTime,varName_all::AbstractVector,Times2read;grdIn=copy(comData.pInfo.grdNaq), datFileFun=comData.pInfo.foreFileFun,varInCtl="data")
    dataOut=Array{Float32,4}(undef,grdIn.lonlatsize...,length(varName_all),length(Times2read))
    Threads.@threads for i in eachindex(varName_all)
        varName=varName_all[i]
        readFore_aStart!(view(dataOut,:,:,i:i,:),tStart,varName,Times2read;grdIn=copy(grdIn),datFileFun=datFileFun,varInCtl=varInCtl)
    end
    # end
    return dataOut
end
function readFore_aStart!(data2read,tStart::ZonedDateTime,varName::AbstractString,Times2read;grdIn=copy(comData.pInfo.grdNaq), datFileFun=comData.pInfo.foreFileFun,varInCtl="data",comData=comData)
    
    # deal with unusual varName -------------------------------------
    if uppercase(varName)=="LAND"
        data2read .= comData.land
        return data2read
    elseif uppercase(varName)=="TER"
        data2read .= comData.ter
        return data2read
    elseif uppercase(varName)=="LEADTIME"
        leadTimes=Dates.value.(Dates.Hour.(Times2read .- tStart))
        data2read .= reshape(leadTimes,1,1,1,:)
        return data2read
    elseif uppercase(varName)=="HOLIDAY"
        idx=indexin(floor.(astimezone.(Times2read,tz"Asia/Shanghai"),Day),comData.df_holiday.date)
        data2read .= reshape(comData.df_holiday.holidays[idx],1,1,1,:)
        return data2read
    end

    # deal with usual varName -------------------------------------
    grdIn.datFileName=datFileFun(tStart,varName)
    leads_currStart=Dates.value.(Hour.(Times2read .- tStart))
    nZ_range=(minimum(leads_currStart):maximum(leads_currStart)) .+ 1
    if isfile(grdIn.datFileName)
        #println("Reading from ", grdIn.datFileName," for ",Dates.format(Times2read[1],"yyyymmddHH"), " to ",Dates.format(Times2read[end],"yyyymmddHH"))
         return read!(data2read,grdIn,varInCtl,nZ_range)
    else
        @warn "File $(grdIn.datFileName) does not exist"
        return nothing
    end
end

function readFore_aStart(tStart::ZonedDateTime,varName::AbstractString,Times2read;grdIn=copy(comData.pInfo.grdNaq), datFileFun=comData.pInfo.foreFileFun,varInCtl="data",comData=comData)
        
    dataOut=Array{Float32,4}(undef,grdIn.lonlatsize...,1,length(Times2read))
    readFore_aStart!(dataOut,tStart,varName,Times2read;grdIn=grdIn, datFileFun=datFileFun,varInCtl=varInCtl)
end

function writeAnalysis_aStart(data2wirte,tStart::ZonedDateTime,pollName::AbstractString,varInCtl::AbstractString,Times2write;
    grdOut=copy(comData.pInfo.grdRe), datFileFun=comData.pInfo.reFileFun)

    grdOut.datFileName=datFileFun(tStart,pollName)
    leads_currStart=Dates.value.(Hour.(Times2write .- tStart))
    nZ_range=(minimum(leads_currStart):maximum(leads_currStart)) .+ 1
    write(grdOut,data2wirte,varInCtl,nZ_range)
end

function writeFore_aStart(data2wirte,tStart::ZonedDateTime,varName::AbstractString,Times2write;grdOut=copy(comData.pInfo.grdNaq), datFileFun=comData.pInfo.foreFileFun)
    grdOut.datFileName=datFileFun(tStart,varName)
    leads_currStart=Dates.value.(Hour.(Times2write .- tStart))
    nZ_range=(minimum(leads_currStart):maximum(leads_currStart)) .+ 1
    write(grdOut,data2wirte,"data",nZ_range)
end

# ================================== readAnalysis =============================
function readAnalysis(Times2read::AbstractArray{ZonedDateTime},pollName::AbstractString;startLead=0,grdre=copy(comData.pInfo.grdRe),datFileFun=comData.pInfo.reFileFun)
    analysis=readInitBeforeStart(Times2read,pollName;grdIn=grdre,datFileFun=datFileFun,startLead=startLead,varInCtl="data")
    return  analysis
end
function readAnalysis_aStart(tStart::ZonedDateTime,pollName::AbstractString,Times2read::AbstractArray{ZonedDateTime};grdre=copy(comData.pInfo.grdRe),datFileFun=comData.pInfo.reFileFun)
    analysis=readFore_aStart(tStart,pollName,Times2read;grdIn=grdre, datFileFun=datFileFun,varInCtl="data")
    return  analysis
end
function readValidGrid(tStart::ZonedDateTime,pollName::AbstractString,Times2read::AbstractArray{ZonedDateTime};grdre=copy(comData.pInfo.grdRe),datFileFun=comData.pInfo.reFileFun,distThreadhould=comData.pInfo.validLength)
    miniDist_data=readFore_aStart(tStart,pollName,Times2read;grdIn=grdre, datFileFun=datFileFun,varInCtl="miniDist")
    @views miniDist_data[:,:,:,2:end] = max.(miniDist_data[:,:,:,1:end-1],miniDist_data[:,:,:,2:end])
    q_validGrid=miniDist_data .< distThreadhould
    return  q_validGrid
end
function readMiniDist(tStart::ZonedDateTime,pollName::AbstractString,Times2read::AbstractArray{ZonedDateTime};grdre=copy(comData.pInfo.grdRe),datFileFun=comData.pInfo.reFileFun)
    miniDist_data=readFore_aStart(tStart,pollName,Times2read;grdIn=grdre, datFileFun=datFileFun,varInCtl="miniDist")
    @views miniDist_data[:,:,:,2:end] = max.(miniDist_data[:,:,:,1:end-1],miniDist_data[:,:,:,2:end])
    return  miniDist_data
end
readValidGrid_aStart=readValidGrid
# ========================================== dataLoader ========================================== 
mutable struct dataInfoCell
    projetInfo::NamedTuple
    tStart_all::AbstractVector{ZonedDateTime}
    leads2init::AbstractVector{Period}
    leads2Fore::AbstractVector{Period}
    normLayer::NormPollCell
    xVar_STinit::AbstractVector{AbstractString}
    xVar_STfore::AbstractVector{AbstractString}
    xVar_CSfore::AbstractVector{AbstractString}
    oVar_Fore::AbstractVector{AbstractString}
    oVar_Init::AbstractVector{AbstractString}
    yVarName::AbstractString
end

function MLUtils.getobs(info::dataInfoCell, idx::AbstractVector)
    if length(idx) == 1
        data=getobs(info,idx[1])
        function addADim(dataIn)
            if isnothing(dataIn)
                dataOut = dataIn
            else
                dataOut=reshape(dataIn,size(dataIn)...,1)
            end
            return dataOut
        end
        tsk_X_STinit= Threads.@spawn addADim(data.X_STinit)
        tsk_X_STfore= Threads.@spawn addADim(data.X_STfore)
        tsk_X_CSfore= Threads.@spawn addADim(data.X_CSfore)
        tsk_other_Fore= Threads.@spawn addADim(data.other_Fore)
        tsk_other_Init= Threads.@spawn addADim(data.other_Init)
        tsk_Y_fore= Threads.@spawn addADim(data.Y_fore)
        tsk_Y_isValid= Threads.@spawn addADim(data.Y_isValid)

        batchData=(X_STinit=fetch(tsk_X_STinit) , X_STfore=fetch(tsk_X_STfore) ,  X_CSfore=fetch(tsk_X_CSfore) ,Y_fore=fetch(tsk_Y_fore) ,Y_isValid=fetch(tsk_Y_isValid) ,
                    other_Fore=fetch(tsk_other_Fore),other_Init=fetch(tsk_other_Init))
        return batchData
    end

    sample=readFore_aStart(info.tStart_all[1], info.yVarName, info.tStart_all[1:1])
    lonlatsize = size(sample)[1:2];
    T=eltype(sample)
    # pre allocate memory
    Y_fore=   Array{T,5}(undef,lonlatsize...,1,length(info.leads2Fore),length(idx))
    Y_isValid= Array{Bool,5}(undef,lonlatsize...,1,length(info.leads2Fore),length(idx))
    X_STinit=  isempty(info.xVar_STinit) ? nothing : Array{T,5}(undef,lonlatsize...,length(info.xVar_STinit),length(info.leads2init),length(idx))
    X_STfore=  isempty(info.xVar_STfore) ? nothing : Array{T,5}(undef,lonlatsize...,length(info.xVar_STfore),length(info.leads2Fore),length(idx))
    X_CSfore=  isempty(info.xVar_CSfore) ? nothing : Array{T,5}(undef,lonlatsize...,length(info.xVar_CSfore),length(info.leads2Fore),length(idx))
    other_Fore= isempty(info.oVar_Fore) ? nothing : Array{T,5}(undef,lonlatsize...,length(info.oVar_Fore),length(info.leads2Fore),length(idx))
    other_Init= isempty(info.oVar_Init) ? nothing : Array{T,5}(undef,lonlatsize...,length(info.oVar_Init),length(info.leads2init),length(idx))
    # Threads.@threads     
    for n in eachindex(idx)
        data=getobs(info,idx[n])
        Y_fore[:,:,:,:,n]=data.Y_fore
        Y_isValid[:,:,:,:,n]=data.Y_isValid
        if !isempty(info.xVar_STinit)
            X_STinit[:,:,:,:,n]=data.X_STinit
        end
        if !isempty(info.xVar_STfore)
            X_STfore[:,:,:,:,n]=data.X_STfore
        end
        if !isempty(info.xVar_CSfore)
            X_CSfore[:,:,:,:,n]=data.X_CSfore
        end
        if !isempty(info.oVar_Fore)
            other_Fore[:,:,:,:,n]=data.other_Fore
        end
        if !isempty(info.oVar_Init)
            other_Init[:,:,:,:,n]=data.other_Init
        end
    end
    batchData=(X_STinit=X_STinit,  X_STfore=X_STfore, 
                X_Dfore=X_Dfore,  X_CSfore=X_CSfore,
                other_Fore=other_Fore,  other_Init=other_Init,
                Y_fore=Y_fore,   Y_isValid=Y_isValid)
    return batchData
end
function MLUtils.getobs(info::dataInfoCell, idx::Number)
    # Initialize X and Y as empty dictionaries
    pInfo=info.projetInfo
    # prepare infomation ------------
    tStart=info.tStart_all[idx] 
    # println("Reading tStart of ",tStart)
    Times_init=tStart.+info.leads2init;
    Times_fore=tStart.+ info.leads2Fore;
    pollName=replace(info.yVarName, "erd_" => "", "_Err" => "", "_Ana" => "")
    # read data -------------------
    tsk_X_STinit =Threads.@spawn info.normLayer(
        readInitBeforeStart(Times_init, info.xVar_STinit; startLead=0, disp=false, grdIn=pInfo.grdNaq, datFileFun=pInfo.foreFileFun),
        info.xVar_STinit)
    tsk_Y_isValid= Threads.@spawn readValidGrid(tStart,pollName,Times_fore;distThreadhould=pInfo.validLength,grdre=pInfo.grdRe,datFileFun=pInfo.reFileFun)

    tsk_X_STfore= Threads.@spawn  isempty(info.xVar_STfore) ?   nothing :  info.normLayer( 
                readFore_aStart(tStart,info.xVar_STfore,Times_fore;grdIn=pInfo.grdNaq, datFileFun=pInfo.foreFileFun),
                info.xVar_STfore)
    tsk_X_CSfore= Threads.@spawn  isempty(info.xVar_CSfore) ?   nothing :  info.normLayer(
                readFore_aStart(tStart,info.xVar_CSfore,Times_fore;grdIn=pInfo.grdNaq, datFileFun=pInfo.foreFileFun),
                info.xVar_CSfore)
    if endswith(info.yVarName,"_Ana")
        pollName=replace(info.yVarName, "_Ana" => "")
        # println("readAnalysis_aStart")
        tsk_Y_fore =  Threads.@spawn readAnalysis_aStart(tStart, pollName, Times_fore; grdre=pInfo.grdRe, datFileFun=pInfo.reFileFun)  # ******************
    else
        # println("readFore_aStart")
        tsk_Y_fore =  Threads.@spawn readFore_aStart(tStart, info.yVarName, Times_fore; grdIn=pInfo.grdNaq, datFileFun=pInfo.foreFileFun)
    end
    tsk_other_Fore = Threads.@spawn  isempty(info.oVar_Fore) ?   nothing :  readFore_aStart(tStart,info.oVar_Fore,Times_fore;grdIn=pInfo.grdNaq, datFileFun=pInfo.foreFileFun)
    tsk_other_Init = Threads.@spawn isempty(info.oVar_Init) ?   nothing :  readInitBeforeStart(Times_init, info.oVar_Init; startLead=0, disp=false, grdIn=pInfo.grdNaq, datFileFun=pInfo.foreFileFun)

    # for tsk in tsk_all;wait(tsk);end
    # norm and reshape data -------
    data=(#X_STinit=X_STinit,
        X_STinit=fetch(tsk_X_STinit),
        X_STfore=fetch(tsk_X_STfore),
        X_CSfore=fetch(tsk_X_CSfore),
        other_Fore=fetch(tsk_other_Fore), 
        other_Init=fetch(tsk_other_Init), 
        Y_fore=fetch(tsk_Y_fore),
        Y_isValid=fetch(tsk_Y_isValid))

    return data
end
function MLUtils.numobs(info::dataInfoCell)
    return length(info.tStart_all)
end
Base.length(info::dataInfoCell)=numobs(info)
Base.getindex(info::dataInfoCell, idx)=getobs(info,idx)

# ========================================== aLoader ========================================== 
mutable struct asyncLoaderCell
    batchIdx_all::AbstractArray
    dataInfo::dataInfoCell
    nextBatch::Int
    batchSize::Int
    shuffle::Bool
    partial::Bool
    isBuffer::Bool
    dataProcessor::Any
    task::Task
end
function asyncLoaderCell(dataInfo::dataInfoCell;batchsize=1,shuffle=false,dataProcessor=x->x,partial=true,isBuffer=false)
    t=Threads.@spawn nothing
    aLoader=asyncLoaderCell([],dataInfo,1,batchsize,shuffle,partial,isBuffer,dataProcessor,t)
    aLoader=reset!(aLoader)
    return aLoader
end

function reset!(aLoader::asyncLoaderCell)
    aLoader.nextBatch=1
    aLoader.batchIdx_all=collect(eachobs(1:length(aLoader.dataInfo), batchsize=aLoader.batchSize,shuffle=aLoader.shuffle,partial=aLoader.partial))
    if aLoader.partial == false  && length(aLoader.batchIdx_all) > 1 && length(aLoader.batchIdx_all[end]) != aLoader.batchSize
        aLoader.batchIdx_all=batchIdx_all[1:end-1]
    end
    if aLoader.isBuffer
        currLoader=aLoader
        aLoader.task=Threads.@spawn currLoader.dataProcessor(currLoader.dataInfo[currLoader.batchIdx_all[currLoader.nextBatch]])
    end
    return aLoader
end

function next!(aLoader::asyncLoaderCell)
    dataCurr=fetch(aLoader.task)
    if aLoader.nextBatch < length(aLoader.batchIdx_all)
        aLoader.nextBatch+=1
        currLoader=aLoader
        aLoader.task=Threads.@spawn currLoader.dataProcessor(currLoader.dataInfo[currLoader.batchIdx_all[currLoader.nextBatch]])
    else# prepare the next epoch
        reset!(aLoader)
    end
    return dataCurr
end

function next(aLoader::asyncLoaderCell)
    dataCurr=aLoader.dataProcessor(aLoader.dataInfo[aLoader.batchIdx_all[aLoader.nextBatch]])
    if aLoader.nextBatch < length(aLoader.batchIdx_all)
        aLoader.nextBatch += 1  
    else 
        reset!(aLoader)
    end
    return dataCurr
end

Base.length(aLoader::asyncLoaderCell)=length(aLoader.batchIdx_all)

mutable struct CuLoader
    loader::Base.Iterators.Stateful
    task::Task
end
function CuLoader(loader::DataLoader)
    loader=Iterators.Stateful(loader)
    task=Threads.@spawn gpu(iterate(loader))
    CuLoader(loader,task)
end
function Base.iterate(loader::CuLoader)
    dataCurr,_=fetch(loader.task)
    if length(loader.loader) > 0
        loader.task=Threads.@spawn gpu(iterate(loader.loader))
    else
        loader.task=Threads.@spawn nothing,nothing
    end
    return dataCurr
end
Base.length(loader::CuLoader)=length(loader.loader)
# ================================== testing =============================
