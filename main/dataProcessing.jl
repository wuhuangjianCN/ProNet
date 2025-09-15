using Revise, ProNetM
using MyPlotM, StaDataM, Dates, Statistics,utilities,CSV,DataFrames,TimeZones,NaNStatistics

raw"""
pollName_all=["PM25"]
pInfo=getProjectInfo()
tStart_all=ZonedDateTime.(DateTime(2023,1,1,12):Day(1):DateTime(2023,1,31,12),tz"UTC");
fileFun_In(tStart,varName)="$(pInfo.dataFolder)/outputs/STCS/$varName.$(Dates.format(tStart,"yyyymmddHH")).dat"
extractStaByStart(tStart_all,pollName_all,fileFun_In;disp=true)
fileFun_In(tStart,varName)="$(pInfo.dataFolder)/Naqpms/$(Dates.format(tStart,"yyyymmddHH"))/$varName.$(Dates.format(tStart,"yyyymmddHH")).dat"
extractStaByStart(tStart_all,pollName_all,fileFun_In;disp=true)
"""
function extractStaByStart(tStart_all,pollName_all,fileFun_In;disp=true,force_overwrite=false)
    if pollName_all isa AbstractString; pollName_all=[pollName_all];end
    staInfo=ProNetM.getStaInfo("staInfo_nation.csv")
    lonM,latM=getLonLatM()
    obs_mapper=obsMapper(lonM,latM,staInfo.lon,staInfo.lat;method="toGrid")
    for pollName in pollName_all
        Threads.@threads for n_t in eachindex(tStart_all)
            tStart=tStart_all[n_t]
            fileIn=fileFun_In(tStart,pollName)
            fileOut=splitext(fileIn)[1] *".mat"
            # skip if output file is exist and newer 
            if !force_overwrite 
                if isfile(fileOut)
                    if  mtime(fileOut) > mtime(fileIn)
                        if disp;println("Skip newer: ",fileOut);end
                        continue
                    end
                end
            end
        
            if disp;println("Extract and writing ",fileOut);end
            grdNaq=getProjectInfo()[:grdNaq]
            grdNaq.datFileName=fileIn
            meshData=read(grdNaq,"data")
            staDataAll=permutedims(mapslices(x->map2sta(obs_mapper,x),meshData,dims=(1,2)),(3,2,1))
            # println(size(staDataAll))
            Times=tStart .+ Hour.(0:size(meshData,3)-1)
            staData=staDATA(Times,pollName,staInfo,staDataAll)
            write2mat(staData,fileOut)
        end
    end
end
function extractStaFromMesh(Times,pollName,meshData;staInfo=ProNetM.getStaInfo("staInfo_nation.csv"))
    lonM,latM=getLonLatM()
    dims2drop=Tuple(findall(size(meshData).==1))
    meshData=dropdims(meshData,dims=dims2drop)
    obs_mapper=obsMapper(lonM,latM,staInfo.lon,staInfo.lat;method="toGrid")
    staDataAll=permutedims(mapslices(x->map2sta(obs_mapper,x),meshData,dims=(1,2)),(3,2,1))
    staObj=staDATA(Times,pollName,staInfo,staDataAll)
    return staObj
end
function mergeStaFore_byMonth(tStart_all,fileFun_in,fileFun_out;force_overwrite=false,disp=true)
    tMonth_all=unique(floor.(astimezone.(tStart_all,tz"UTC"),Month))
    for tMonth in tMonth_all
        # tStart_all_curr=filter(t->floor(t,Month)==tMonth,tStart_all)
        tStart_all_aMonth=(tMonth:Day(1):tMonth+Month(1)-Hour(1)) .+Hour(12)
        mergeStaFore(tStart_all_aMonth,fileFun_in,fileFun_out(tMonth);force_overwrite=force_overwrite,disp=disp)
    end
end
function mergeStaFore(tStart_all,fileFun_in,fileName_out;force_overwrite=false,disp=true)
    # geth staTmp for variable init, and inFile_mtime for skiping
    initFileName=nothing
    inFile_mtime=[]
    for tStart in tStart_all
        if isfile(fileFun_in(tStart))
            if isnothing(initFileName)
                initFileName=fileFun_in(tStart)
            end
            inFile_mtime=push!(inFile_mtime, mtime(fileFun_in(tStart)))
        end
    end

    # skip if output file is exist and newer
    if  !force_overwrite
        if  isfile(fileName_out) 
            if  mtime(fileName_out) > maximum(inFile_mtime) 
                if disp;println("Skip newer: ",fileName_out);end
                return 
            end
        end
    end

    # merge data
    if disp;        println("============ Making ",fileName_out," =============");    end
    staTmp=staDATA(initFileName)
    staInfo=staTmp.staInfo
    leads_all=collect(0:length(staTmp.times)-1)
    varName_all=staTmp.varNames
    staDataAll=fill(NaN,length(tStart_all),length(varName_all),length(staInfo.staID),length(staTmp.times))
    Threads.@threads for n_start  in eachindex(tStart_all)
        tStart = tStart_all[n_start]
        file_curr=fileFun_in(tStart)        
        if !isfile(file_curr)
            @warn "file not exist: $file_curr"
            continue
        else
            if disp;println("Processing ",file_curr);end
        end
        sta_curr=toForeForm(staDATA(file_curr),tStart,leads_all)
        staDataAll[findfirst(tStart.==tStart_all),:,:,:]=read(sta_curr;Times=tStart,varNames=varName_all,staIDs=staInfo.staID)
    end
    staMerge=staDATA(tStart_all,varName_all,staInfo,leads_all,staDataAll)
    write2mat(staMerge,fileName_out)
end
function unifyCityName(city_all)    
    city_all=replace.(city_all,"市"=>"","地区"=>"","区"=>"","自治州"=>"","州"=>"")
    map!(s ->length(s)== 1 ? s * "州" : s, city_all, city_all)
    replace!(city_all,"雄安新"=>"雄安新区")
    return city_all
end
# ========================= rewrite plot_getData4HumanvsAI ============================
# merge staInfo to cityInfo, calculate the lon, lat by the mean of stations in the city

function mergeCity(staObj;Times=staObj.times,varNames=staObj.varNames,staInfo=staObj.staInfo,city_all=unique(staInfo.city),leads_all=staObj.leadTimes)
    if !(varNames isa AbstractVector);varNames=[varNames];end
    cityInfo=mergeStaInfo_ByCity(staInfo;city_all=city_all)
    # exclude 对照点
    q_valid=.!occursin.("(对照点)",staInfo.staName)
    staInfo=staInfo[q_valid,:]
    staDataAll=fill(NaN,length(Times),length(varNames),length(city_all),length(leads_all))
    for (n_city,city) in enumerate(city_all)
        staIDs_aCity=staInfo.staID[isequal.(staInfo.city,city)]
        staDataAll[:, :, n_city:n_city,:]=nanmean(read(staObj;Times=Times,varNames=varNames,staIDs=staIDs_aCity,leadTimes=leads_all); dims=3)
    end
    cityData=staDATA(Times,varNames,cityInfo,leads_all,staDataAll)
    return cityData
end
function mergeStaInfo_ByCity(staInfo;city_all=unique(staInfo.city))
    q_invalid=isempty.(city_all)
    city_all=string.(city_all[.!q_invalid])
    
    cityInfo=DataFrame(lon=NaN,lat=NaN,staID=city_all)
    for (n_city,city) in enumerate(city_all)
        q_currCity=isequal.(staInfo.city,city)
        cityInfo.lon[n_city]=nanmean(staInfo.lon[q_currCity])
        cityInfo.lat[n_city]=nanmean(staInfo.lat[q_currCity])
    end
    cityInfo.staID=unifyCityName(cityInfo.staID)
    return cityInfo
end

# merge staDATA to cityDATA, change the staObj format from forecasting form to simulation form, calcuate the day mean and city mean of original data
function mergeCityDay_toSimForm(staFore,tDay_all;varNames=staFore.varNames,city_all=unique(staFore.staInfo.city),
           startLead_all=4 .+ 24 .*(0:6))
    if !(varNames isa AbstractVector) ;varNames=[varNames];end
    staInfo=staFore.staInfo
    staID_all=staInfo.staID[in.(staInfo.city,[city_all])]
    staDataAll=fill(NaN,length(tDay_all),length(varNames),length(city_all),length(startLead_all))
    for (nDayFore,startLead) in enumerate(startLead_all)
        staSim=toSimForm(staFore,startLead;varNames=varNames,staIDs=staID_all)
        staCityDay_currLead=mergeCityDay(staSim;city_all=city_all,tDay_all=tDay_all)        
        staDataAll[:, :, :, nDayFore]=staCityDay_currLead.staDataAll
    end
    cityInfo=mergeStaInfo_ByCity(staInfo;city_all=city_all)
    staCityDay=staDATA(tDay_all,varNames,cityInfo,startLead_all .- minimum(startLead_all),staDataAll)
    return staCityDay
end
function mergeCityDay(staSim;tDay_all=unique(floor.(staSim.times,Day)),staInfo=staSim.staInfo, city_all=unique(staInfo.city), pollName_all=staSim.varNames)

    q_invalid=isempty.(city_all)    
    city_all=string.(city_all[.!q_invalid])
    # exclude 对照点
    q_valid=.!occursin.("(对照点)",staInfo.staName)
    staInfo=staInfo[q_valid,:]
    Times_all=tDay_all[1]:Hour(1):(tDay_all[end]+Hour(23))
    staDataAll=fill(NaN,length(tDay_all),length(pollName_all),length(city_all))
    for (n_city,city) in enumerate(city_all)
        staIDs_aCity=staInfo.staID[isequal.(staInfo.city,city)]
        data_aCity=read(staSim;Times=Times_all,staIDs=staIDs_aCity)
        staDataAll[:, :, n_city]=nanmean(
            reshape(data_aCity,24,length(tDay_all),length(pollName_all),length(staIDs_aCity));
            dims=(4,1))
    end
    cityInfo=mergeStaInfo_ByCity(staInfo;city_all=city_all)
    staCityDay=staDATA(tDay_all,pollName_all,cityInfo,[0],staDataAll)
    return staCityDay
end

function mergeCityOzone_toSimForm(staFore,tDay_all;varNames=staFore.varNames,city_all=unique(staFore.staInfo.city),
    startLead_all=4 .+ 24 .*(0:6))
    if !(varNames isa AbstractVector) ;varNames=[varNames];end
    staInfo=staFore.staInfo
    staID_all=staInfo.staID[in.(staInfo.city,[city_all])]
    staDataAll=fill(NaN,length(tDay_all),length(varNames),length(city_all),length(startLead_all))
    for (nDayFore,startLead) in enumerate(startLead_all)
        staSim=toSimForm(staFore,startLead;varNames=varNames,staIDs=staID_all)
        staCityDay_currLead=mergeCityOzone(staSim;city_all=city_all,tDay_all=tDay_all)        
        staDataAll[:, :, :, nDayFore]=staCityDay_currLead.staDataAll
    end
    cityInfo=mergeStaInfo_ByCity(staInfo;city_all=city_all)
    staCityDay=staDATA(tDay_all,varNames,cityInfo,startLead_all .- minimum(startLead_all),staDataAll)
    return staCityDay
end

function mergeCityOzone(staSim;tDay_all=unique(floor.(staSim.times,Day)),staInfo=staSim.staInfo, city_all=unique(staInfo.city), pollName_all=staSim.varNames)
    
    q_invalid=isempty.(city_all)    
    city_all=string.(city_all[.!q_invalid])
    # exclude 对照点
    q_valid=.!occursin.("(对照点)",staInfo.staName)
    staInfo=staInfo[q_valid,:]
    Times_all=tDay_all[1]:Hour(1):(tDay_all[end]+Hour(23))
    staDataAll=fill(NaN,length(tDay_all),length(pollName_all),length(city_all))
    Threads.@threads for n_city in eachindex(city_all)
        city=city_all[n_city]
        staIDs_aCity=staInfo.staID[isequal.(staInfo.city,city)]
        data_aCity=read(staSim;Times=Times_all,staIDs=staIDs_aCity)
        if size(data_aCity,2) > 1 || !isequal(uppercase.(pollName_all[1]),"O3")
            @error "Only support for 1 variable of Ozone"
        else
            # calculate 8 hour average
            data_aCity_A8H=similar(data_aCity)
            for n_sta in axes(data_aCity,3)
                data_aCity_A8H[:,1,n_sta]=movmean(data_aCity[:,1,n_sta],8)  
            end
            # calculate maximum of 8 hour average in a day
            data_aCity_A8H_Day=nanmaximum(
                    reshape(data_aCity_A8H,24,length(tDay_all),length(staIDs_aCity));
                    dims=1)
            # calculate city mean
            data_aCity_A8H_Day_City=vec(nanmean(data_aCity_A8H_Day;dims=3))
            staDataAll[:, 1, n_city]=data_aCity_A8H_Day_City
        end
    end
    cityInfo=mergeStaInfo_ByCity(staInfo;city_all=city_all)
    staCityDay=staDATA(tDay_all,pollName_all,cityInfo,[0],staDataAll)
    return staCityDay
end

function readCityData_Human(fileName)
    staHuman=staDATA(fileName)
    staHuman.staInfo.staID=replace.(staHuman.staInfo.staID,"市"=>"","地区"=>"","区"=>"","自治州"=>"","哈萨克"=>"","州"=>""
        ,"朝鲜族"=>"","藏族羌族"=>"","藏族"=>"","彝族"=>"","布依族苗族"=>"",
        "回族"=>"","苗族侗族"=>"","蒙古族藏族"=>"","蒙古"=>"","柯尔克孜"=>"","土家族苗族"=>"")
    staHuman.staDataAll[staHuman.staDataAll .== -99] .= NaN32
    staHuman.staDataAll[staHuman.staDataAll .> 1000] .= NaN32
    staHuman.leadTimes=24 .* (0:(length(staHuman.leadTimes)-1))
    #统一名称
    map!(s ->length(s)==1 ? s * "州" : s, staHuman.staInfo.staID , staHuman.staInfo.staID)
    return staHuman
end

function readCityData_SimForm(fileName_all;Times=nothing,varNames=nothing,staInfo=nothing)
    staObj = staDATA(fileName_all;Times=Times,varNames=varNames,staInfo=staInfo)
    staObj = mergeCityDay(staObj)
    return staObj
end
function readCityData_ForeForm(fileName_all,tStart_all,tDay_all;varNames=nothing,staInfo=nothing,startLead_all=4 .+ 24 .*(0:6))
    staFore = staDATA(fileName_all;Times=tStart_all,varNames=varNames,staInfo=staInfo)
    # exclude 对照点
    q_valid=.!occursin.("(对照点)",staFore.staInfo.staName)
    staFore=extract(staFore;staIDs=staFore.staInfo.staID[q_valid])
    staObj =mergeCityDay_toSimForm(staFore,tDay_all;varNames=staFore.varNames,city_all=unique(staInfo.city), startLead_all=startLead_all)
    return staObj
end
function readCityOzone_ForeForm(fileName_all,tStart_all,tDay_all;varNames=nothing,staInfo=nothing,startLead_all=4 .+ 24 .*(0:6))
    staFore = staDATA(fileName_all;Times=tStart_all,varNames=varNames,staInfo=staInfo)
    # exclude 对照点
    q_valid=.!occursin.("(对照点)",staFore.staInfo.staName)
    staFore=extract(staFore;staIDs=staFore.staInfo.staID[q_valid])
    staObj =mergeCityOzone_toSimForm(staFore,tDay_all;varNames=staFore.varNames,city_all=unique(staInfo.city), startLead_all=startLead_all)
    return staObj
end
function getStaInfo2evalute(staInfo=ProNetM.getStaInfo("staInfo_nation.csv"))
    q_invalid=occursin.("(对照",staInfo.staName) .|| ismissing.(staInfo.city)
    staInfo=staInfo[.!q_invalid,:]
    city_sta=replace.(staInfo.city,"市"=>"","地区"=>"","区"=>"","自治州"=>"","州"=>"")
    staInfo.city=city_sta 
    #统一名称
    map!(s ->length(s)== 1 ? s * "州" : s, staInfo.city, staInfo.city)
    replace!(staInfo.city,"雄安新"=>"雄安新区")    
    # exclude 三沙 雄安新区
    staInfo=staInfo[.!in.(staInfo.city ,[["三沙","雄安新区"]]),:]
    return staInfo
end

# staInfo=staInfo[staInfo.province.=="河南",:]
# city2plot=unique(staInfo.city)
function getData4HumanvsAI(model,city2plot,tDay_all,nDayFor_all; 
            pInfo=getProjectInfo(),
            pollName="PM25",
            staInfo=ProNetM.getStaInfo("staInfo_nation.csv")
        )
    tDay_all=astimezone.(tDay_all,tz"Asia/Shanghai")
    Times_all=tDay_all[1]:Hour(1):(tDay_all[end]+Hour(23))

    # 基础站点信息
    staInfo=getStaInfo2evalute(staInfo)

    # 筛选画图城市
    if city2plot != "全国"
        staInfo=staInfo[in.(staInfo.city,[city2plot]),:]
    else
        city2plot=unique(staInfo.city)
    end
    # ====================== 读取人工预报 =======================
    staHuman_mid = readCityData_Human(pInfo.staPM25mid)
    staHuman_obs = readCityData_Human(pInfo.staPM25obs)
        

    # ================= 整理站点观测信息 ======================
    # files_obs=[pInfo.dataFolder * "/Obs/pollObsQC$(Dates.format(t,"yyyymm")).mat" for t in unique(floor.(Times_all,Month))]
    # staObsQC = readCityData_SimForm(files_obs;Times=Times_all,varNames=[pollName],staInfo=staInfo)

    # ============== 读取AI、Naqpms、再分析预报 ===================
    # 计算所有起报时间
    start_time_utc = astimezone(tDay_all[1], tz"UTC") - Day(8) - Hour(4)
    end_time_utc = astimezone(tDay_all[end], tz"UTC") + Day(1) - Hour(4)
    tStart_all = ZonedDateTime.(DateTime(start_time_utc):Day(1):DateTime(end_time_utc), tz"UTC")
    tMonth_all=unique(floor.(tStart_all,Month))    
    startLead_all=4 .+ 24 .*(nDayFor_all .- 1)

    # for AI
    files=[pInfo.dataFolder * "/outputs/$(model)/$pollName.$(Dates.format(t,"yyyymm")).mat" for t in tMonth_all]
    staFore_AI=readCityData_ForeForm(files,tStart_all,tDay_all;varNames=[pollName],staInfo=staInfo,startLead_all=startLead_all)
    # for naq
    files=[pInfo.dataFolder * "/Naqpms/$pollName.$(Dates.format(t,"yyyymm")).mat" for t in tMonth_all]
    staFore_Naq=readCityData_ForeForm(files,tStart_all,tDay_all;varNames=[pollName],staInfo=staInfo,startLead_all=startLead_all)
    # for reanalysis
    # files=[pInfo.dataFolder * "/reanalysis/$pollName/$pollName.$(Dates.format(t,"yyyymm")).mat" for t in tMonth_all]
    # staFore_Re=readCityData_ForeForm(files,tStart_all,tDay_all;varNames=[pollName],staInfo=staInfo,startLead_all=startLead_all)

    # ================== 计算各种指标 ===================
    # get data
    human_all=read(staHuman_mid;Times=tDay_all,varNames=pollName,staIDs=city2plot,leadTimes=24 .* (nDayFor_all .-1))
    obs_all=read(staHuman_obs;Times=tDay_all,varNames=pollName,staIDs=city2plot,leadTimes=24 .* (nDayFor_all .-1))
    ai_all=read(staFore_AI;Times=tDay_all,varNames=pollName,staIDs=city2plot,leadTimes=24 .* (nDayFor_all .-1))
    naq_all=read(staFore_Naq;Times=tDay_all,varNames=pollName,staIDs=city2plot,leadTimes=24 .* (nDayFor_all .-1))
    # calculate rmse compare
    f_rmse(pre,target)=dropdims(sqrt.(nanmean((pre-target).^2;dims=3)),dims=(2,3))
    humanVSai= f_rmse(human_all,obs_all) - f_rmse(ai_all,obs_all)
    humanVSnaq= f_rmse(human_all,obs_all) - f_rmse(naq_all,obs_all)
    naqpmsVSai= f_rmse(naq_all,obs_all) - f_rmse(ai_all,obs_all)
    # calculate city mean
    human_cityMean=dropdims(nanmean(human_all,dims=3);dims=(2,3))
    obs_cityMean=dropdims(nanmean(obs_all,dims=3);dims=(2,3))
    ai_cityMean=dropdims(nanmean(ai_all,dims=3);dims=(2,3))

    return humanVSai, humanVSnaq, human_cityMean, obs_cityMean, ai_cityMean, naqpmsVSai
end
