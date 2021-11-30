%% setting
PollFore='O3';
%PollFore='PM25';
lossRadius=50e3;
predictingRadius=150e3;
startHour=0;
testDataFolder='F:\data_proNet';
trainDataFolder='C:\data_proNet';

winterMonth_training=datetime([2017 1 1],'TimeZone','Asia/Shanghai');
winterMonth_forecast=datetime([2020 1 1],'TimeZone','Asia/Shanghai');
summerMonth_training=datetime([2017 7 1],'TimeZone','Asia/Shanghai');
summerMonth_forecast=datetime([2020 7 1],'TimeZone','Asia/Shanghai');

tStarts_training=(winterMonth_training-days(6)):days(1):(dateshift(winterMonth_training,'start','month','next')-days(1));
tStarts_training=[tStarts_training,(summerMonth_training-days(7)):days(1):(dateshift(summerMonth_training,'start','month','next')-days(1))];
tStarts_training.TimeZone='Asia/Shanghai';tStarts_training.Hour=startHour;
tStarts_forecast=(winterMonth_forecast-days(6)):days(1):(dateshift(winterMonth_forecast,'start','month','next')-days(1));
tStarts_forecast=[tStarts_forecast,(summerMonth_forecast-days(7)):days(1):(dateshift(summerMonth_forecast,'start','month','next')-days(1))];
tStarts_forecast.TimeZone='Asia/Shanghai';tStarts_forecast.Hour=startHour;

addpath('ProNet_allInOne')
tag_experiment=['proNet_' PollFore '_stcs_paper'];
outDataFolder=fullfile('F:\','data_proNet_paper',tag_experiment);
if ~exist(outDataFolder,'dir');mkdir(outDataFolder);end
timerTotal=tic;
figureFolder=fullfile(pwd,tag_experiment);
if ~exist(figureFolder,'dir');mkdir(figureFolder);end

% prepare the meshDATA of input by Start ---------------------------
customOrder=[1 2 5 6 4 3];% [lon lat var start lead]
inNCFileNameFun=@(t,tag) fullfile(trainDataFolder,['dataByStart' datestr(t,'yyyymmdd') '_' tag '.nc']);
meshObj_ForeTrainCell=cell(size(tStarts_training));   meshObj_InitTrainCell=cell(size(tStarts_training));
for n_t=1:numel(tStarts_training)
    tStart_current=tStarts_training(n_t);
    fileName=inNCFileNameFun(tStart_current,'fore');    meshObj_ForeTrainCell{n_t} = meshDATA(fileName);     meshObj_ForeTrainCell{n_t}.setCustomDimOrder(customOrder);
    fileName=inNCFileNameFun(tStart_current,'init');    meshObj_InitTrainCell{n_t} = meshDATA(fileName);     meshObj_InitTrainCell{n_t}.setCustomDimOrder(customOrder);
end
leadsAll_forecast=meshObj_ForeTrainCell{1}.leadTimes; leadsAll_init=meshObj_InitTrainCell{1}.leadTimes;

inNCFileNameFun=@(t,tag) fullfile(testDataFolder,['dataByStart' datestr(t,'yyyymmdd') '_' tag '.nc']);
meshObj_ForeTestCell=cell(size(tStarts_forecast));   meshObj_InitTestCell=cell(size(tStarts_forecast));
for n_t=1:numel(tStarts_forecast)
    tStart_current=tStarts_forecast(n_t);
    fileName=inNCFileNameFun(tStart_current,'fore');    meshObj_ForeTestCell{n_t} = meshDATA(fileName);     meshObj_ForeTestCell{n_t}.setCustomDimOrder(customOrder);
    fileName=inNCFileNameFun(tStart_current,'init');    meshObj_InitTestCell{n_t} = meshDATA(fileName);     meshObj_InitTestCell{n_t}.setCustomDimOrder(customOrder);
end
% combine the tran and test meshData
tStarts_all=[tStarts_training,tStarts_forecast];
meshObj_ForeAllCell=[meshObj_ForeTrainCell(:) ;meshObj_ForeTestCell(:) ];
meshObj_InitAllCell=[meshObj_InitTrainCell(:) ;meshObj_InitTestCell(:) ];

%  prepare output fileNameFun --------------------------
postFileFun=@(t) fullfile(outDataFolder,['DSpost_' PollFore '_' datestr(t,'yyyymmdd')  '.nc']); %
postCombineFileName=fullfile(outDataFolder,['DSpost_' PollFore '.nc']); %
objCurr=meshObj_ForeTestCell{1};  nZs=objCurr.nZs;  sizeLonLat=objCurr.sizeLonLat;
meshObj_DSpostAllCell=cell(size(tStarts_all));
for n_t=1:numel(tStarts_all)
    tStart_current=tStarts_all(n_t);
    fileName=postFileFun(tStart_current);    
    meshObj_DSpostAllCell{n_t} = meshDATA(fileName,sizeLonLat,nZs,leadsAll_forecast,{'ConPred','ConPred_st','ConPred_ietm','ConPred_erd','stErrPred','csErrPred','bErrPred','errAllPred'},tStart_current);
    meshObj_DSpostAllCell{n_t}.setCustomDimOrder(customOrder);
end
lia=ismember(tStarts_all,tStarts_training);
meshObj_DSpostTrainCell=meshObj_DSpostAllCell(lia);
lia=ismember(tStarts_all,tStarts_forecast);
meshObj_DSpostTestCell=meshObj_DSpostAllCell(lia);


load staInfoAll.mat staInfo 
load lonMlatM_d2.mat lonM latM

% init_network --------------------------------------
sizeGrid=size(lonM);
%%  training area
[~,minDist]=findNearest(staInfo.lon,staInfo.lat,lonM(:),latM(:));
q_select=minDist< predictingRadius;
gridWeight_predict=reshape(double(q_select),size(lonM));
[~,minDist]=findNearest(staInfo.lon,staInfo.lat,lonM(:),latM(:));
q_select=minDist< lossRadius;
gridWeight_loss=reshape(double(q_select),size(lonM));

%% ### Training model_st #########################################################################################################

% xVarInfo  varName_init varName_forecast
xVarName_st={ ['erd_' PollFore 'Err_anHourBefore'] ['erd_' PollFore] PollFore };

% ------------------------ prepare data ---------------------------------

% training data --------------
folderTmp=fullfile(trainDataFolder,tag_experiment);if ~exist(folderTmp,'dir');mkdir(folderTmp);end
fileNameFun_trainingData=@(name) fullfile(folderTmp,name);
fileName_dlX_stInit=fileNameFun_trainingData('train_dlX_stInit.nc');
fileName_dlX_st=fileNameFun_trainingData('train_dlX_st.nc');
fileName_dlY_stTarget=fileNameFun_trainingData('train_dlY_stTarget.nc');
fileName_stLossWeight=fileNameFun_trainingData('train_stLossWeight.nc');
if ~exist(fileName_dlX_st,'file')
    prepareMesh_modelST(PollFore,xVarName_st,lossRadius,fileName_dlX_stInit,fileName_dlX_st,fileName_dlY_stTarget,fileName_stLossWeight,tStarts_training,meshObj_InitTrainCell,meshObj_ForeTrainCell)
end
% testing data --------------
folderTmp=fullfile(testDataFolder,tag_experiment);if ~exist(folderTmp,'dir');mkdir(folderTmp);end
fileNameFun_trainingData=@(name) fullfile(folderTmp,name);
testFileName.fileName_dlX_stInit=fileNameFun_trainingData('test_dlX_stInit.nc');
testFileName.fileName_dlX_st=fileNameFun_trainingData('test_dlX_st.nc');
testFileName.fileName_dlY_stTarget=fileNameFun_trainingData('test_dlY_stTarget.nc');
testFileName.fileName_stLossWeight=fileNameFun_trainingData('test_stLossWeight.nc');
if ~exist(testFileName.fileName_dlX_st,'file')
    prepareMesh_modelST(PollFore,xVarName_st,lossRadius,testFileName.fileName_dlX_stInit,testFileName.fileName_dlX_st,testFileName.fileName_dlY_stTarget,testFileName.fileName_stLossWeight,tStarts_forecast,meshObj_InitTestCell,meshObj_ForeTestCell)
end

%% ### Training model_st and model_cs simultanously #########################################################################################################
% xVarName_cs={ 'erd_PM25' 'PM25' 'erd_ANH4' 'erd_BC'  'erd_ANO3' 'erd_ASO4' 'erd_OC'  'erd_SOA'  'temp' 'rh'};
xVarName_cs={'NH3' 'erd_ANH4' 'erd_ANO3' 'HCHO' 'ASO4' 'ANO3'  'ANH4' 'PMC' 'HONO' 'SOA' 'NO2' 'erd_PM25' ...
    'erd_HONO' 'erd_PM10' 'SO2' 'erd_ASO4' 'erd_NH3' 'OC' 'erd_OC' 'erd_HCHO' 'erd_PMC' 'erd_PMF'  'erd_CO' 'PMF' 'NO' ...
    'BC' 'erd_NO2' 'CO' 'C2H6' 'ISOP' 'erd_BC'  'PM25' 'PM10' 'erd_SOA' 'erd_NO' 'erd_C2H6'  'erd_SO2' ...
     'erd_ISOP'  'HNO3' 'erd_O3' 'erd_HNO3' 'O3' ...
     'temp' 'rh'  'CLDOPD'} ;
% xVarName_cs={'NH3' 'erd_ANH4' 'erd_ANO3' 'HCHO' 'ASO4' 'ANO3' 'PAN' 'ANH4' 'PMC' 'ALD2' 'HONO' 'SOA' 'NO2' 'erd_PM25' ...
%     'erd_HONO' 'erd_PM10' 'SO2' 'erd_ASO4' 'erd_PAN' 'erd_NH3' 'OC' 'erd_OC' 'erd_HCHO' 'erd_PMC' 'erd_PMF' 'erd_ALD2' 'erd_CO' 'PMF' 'NO' ...
%     'BC' 'erd_NO2' 'CO' 'erd_ETH' 'ETH' 'C2H6' 'PAR' 'ISOP' 'erd_BC' 'erd_OLE' 'PM25' 'OLE' 'PM10' 'erd_SOA' 'erd_NO' 'erd_C2H6' 'erd_PAR' 'erd_SO2' ...
%     'TOL' 'erd_ISOP' 'XYL' 'erd_XYL' 'HNO3' 'SSA' 'erd_SSA' 'erd_O3' 'erd_HNO3' 'erd_TOL' 'erd_NAP' 'O3' 'NAP'...
%      'temp' 'rh' 'AOD' 'CLDOPD'} ;
varName2portion={};%'ANH4' 'BC'  'ANO3' 'ASO4' 'OC'  'SOA' };
% prepare normalizer_CS and csDataFun ----------------
meanInNormalizer=zeros(size(xVarName_cs));
stdInNormalizer=zeros(size(xVarName_cs));    
leads4norm=1:24;
WaitMessage = parfor_wait(numel(tStarts_training),'Waitbar',true,'Title','normalizer_CS');
for n_t=1:numel(tStarts_training)
    meshObj_curr=meshObj_ForeTrainCell{n_t};
    Xdata_curr=myCSDATAFun(meshObj_curr,xVarName_cs,varName2portion,[],leads4norm);
    q_select=meshObj_curr.get([],(min(leads4norm)-1):max(leads4norm),{[PollFore 'minValidDist']},[])<lossRadius;
    q_select=q_select(:,:,:,:,1:end-1) & q_select(:,:,:,:,2:end);
    meanCurr=nan(size(xVarName_cs));stdCurr=nan(size(xVarName_cs));
    for n_var=1:numel(xVarName_cs)
        data_aVar=Xdata_curr(:,:,n_var,:,:);
        data_aVar=data_aVar(q_select);
        meanCurr(n_var)=mean(data_aVar,'all','omitnan');
        stdCurr(n_var)=std(data_aVar,0,'all','omitnan');
    end
    meanInNormalizer=meanInNormalizer+meanCurr/numel(tStarts_training);
    stdInNormalizer=stdInNormalizer+stdCurr/numel(tStarts_training);
    WaitMessage.Send;
end
    WaitMessage.Destroy;
normalizer_CS=normClass({});normalizer_CS.mean_all=meanInNormalizer(:);normalizer_CS.std_all=stdInNormalizer(:);normalizer_CS.varName_all=xVarName_cs(:);
csDataFun=@(meshObjIn) myCSDATAFun(meshObjIn,xVarName_cs,varName2portion,normalizer_CS);
nLayer=8;nWidth=200;

parameters_cs=init_netCS(numel(xVarName_cs),nLayer,nWidth);
[parameters_st,state_st,config]=init_netST(sizeGrid);
%
paraFolder=outDataFolder;if ~exist(paraFolder,'dir');mkdir(paraFolder);end
for epoch=1:30
    tag_curr=['_epoch' num2str(epoch)];
    clear stcsOption
    stcsOption.epoch=1;stcsOption.plotTraining=true;
    parameters_st=fieldfun(@gpuArray,parameters_st);state_st=fieldfun(@gpuArray,state_st);
    for nCell=1:numel(parameters_cs);parameters_cs{nCell}=fieldfun(@gpuArray,parameters_cs{nCell});end
    [parameters_st,parameters_cs,hf1]=trainModel_STCS_paper(xVarName_st,fileName_dlX_stInit,fileName_dlX_st,fileName_dlY_stTarget,fileName_stLossWeight,parameters_st,state_st,csDataFun,meshObj_ForeTrainCell,parameters_cs,stcsOption);

    savePng(fullfile(tag_experiment,['training loss for model_STCS' tag_curr '.png']),hf1);close(hf1);
    parameters_st=fieldfun(@gather,parameters_st);
    for nCell=1:numel(parameters_cs);parameters_cs{nCell}=fieldfun(@gather,parameters_cs{nCell});end
    save(fullfile(paraFolder,['network_stcs' tag_curr '.mat']),"parameters_st","parameters_cs","normalizer_CS");%,'state_st')
end

%% *** Results using model_st  ***************************************************
disp('********************* Results using model_ST **********************')

xVarName_st={ ['erd_' PollFore 'Err_anHourBefore'] ['erd_' PollFore] PollFore };
parameters_st=fieldfun(@gpuArray,parameters_st);state_st=fieldfun(@gpuArray,state_st);
calAndWriteModel_st(meshObj_DSpostTrainCell,xVarName_st,fileName_dlX_stInit,fileName_dlX_st,parameters_st,state_st)
calAndWriteModel_st(meshObj_DSpostTestCell,xVarName_st,testFileName.fileName_dlX_stInit,testFileName.fileName_dlX_st,parameters_st,state_st)

%%%### train model_CS #############################################################
% %save E:\data2deepNN.mat PollFore lossRadius meshObj_ForeTrainCell meshObj_DSpostTrainCell  tStarts_training meshObj_ForeTestCell meshObj_DSpostTestCell tStarts_forecast
% reset(gpuDevice(1));
% % prepare dlX dlY --------------------load(paraFileName,'parameters_cs','normalizer_CS')
% normPara=normalizer_CS;
% %xVarName_cs={'rh' 'NH3' 'erd_ANH4' 'erd_ANO3' 'HCHO' 'ASO4' 'ANO3' 'PAN' 'ANH4' 'PMC' 'ALD2' 'HONO' 'SOA' 'NO2' 'erd_PM25' ...
% %     'erd_HONO' 'erd_PM10' 'SO2' 'erd_ASO4' 'erd_PAN' 'erd_NH3' 'OC' 'erd_OC' 'erd_HCHO' 'erd_PMC' 'erd_PMF' 'erd_ALD2' 'erd_CO' 'PMF' 'NO' ...
% %     'BC' 'erd_NO2' 'CO' 'erd_ETH' 'ETH' 'C2H6' 'PAR' 'ISOP' 'erd_BC' 'erd_OLE' 'PM25' 'OLE' 'PM10' 'erd_SOA' 'erd_NO' 'erd_C2H6' 'erd_PAR' 'erd_SO2' ...
% %     'TOL' 'erd_ISOP' 'XYL' 'erd_XYL' 'HNO3'  'erd_O3' 'erd_HNO3' 'erd_TOL' 'temp' 'erd_NAP' 'O3' 'NAP' };
% xVarName_cs=normPara.varName_all;
% varS2read=xVarName_cs;%[xVarName_cs  'PM25'];
% varName2portion={};%{'ANH4' 'BC'  'ANO3' 'ASO4' 'OC'  'SOA' };
% % ------------------------ prepare data ---------------------------------
% leads4modelR=1:24;
% %if  ~exist('X4cs_train_extend','var') || size(X4cs_train_extend,1)~= numel(varS2read) 
% if true
%     winterMonth_training=datetime([2017 1 1],'TimeZone','Asia/Shanghai');
%     winterMonth_forecast=datetime([2020 1 1],'TimeZone','Asia/Shanghai');
%     summerMonth_training=datetime([2017 7 1],'TimeZone','Asia/Shanghai');
%     summerMonth_forecast=datetime([2020 7 1],'TimeZone','Asia/Shanghai');
%     Times2evaluate_training=[winterMonth_training:hours(1):(dateshift(winterMonth_training,'start','month','next')-hours(1)),...
%         summerMonth_training:hours(1):(dateshift(summerMonth_training,'start','month','next')-hours(1))];
%     [X4cs_train_extend,dlYtrain_csOri]=prepareData_modelCS(PollFore,varS2read,leads4modelR,lossRadius,meshObj_ForeTrainCell,meshObj_DSpostTrainCell,Times2evaluate_training);
%     
%     Times2evaluate_testing=[winterMonth_forecast:hours(1):(dateshift(winterMonth_forecast,'start','month','next')-hours(1)),...
%         summerMonth_forecast:hours(1):(dateshift(summerMonth_forecast,'start','month','next')-hours(1))];
%     [X4cs_test_extend,dlYtest_csOri]=prepareData_modelCS(PollFore,varS2read,leads4modelR,lossRadius,meshObj_ForeTestCell,meshObj_DSpostTestCell,Times2evaluate_testing);
% end
% dlXtrain_cs=X4cs_train_extend(ismember(varS2read,xVarName_cs),:);
% dlXtest_cs=X4cs_test_extend(ismember(varS2read,xVarName_cs),:);
% normalizer_CS=normClassGeneral({});normalizer_CS.varName_all=normPara.varName_all;
% normalizer_CS.mean_all=normPara.mean_all;normalizer_CS.std_all=normPara.std_all;
% dlXtrain_cs=normalizer_CS.normData(dlXtrain_cs,xVarName_cs);
% dlXtest_cs=normalizer_CS.normData(dlXtest_cs,xVarName_cs);
% 
% % ------------------------ preparing parameters -----------------------------
% parameters_cs=init_netCS(numel(xVarName_cs),nLayer,nWidth);
% for nCell=1:numel(parameters_cs);parameters_cs{nCell}=fieldfun(@gpuArray,parameters_cs{nCell});end
% 
% % ------------------------ train the model -----------------------------
% dlX=dlarray(dlXtrain_cs);dlY_Target=dlarray(dlYtrain_csOri);testData.dlX=dlarray(dlXtest_cs); testData.dlY_Target=dlYtest_csOri;
% %dlX=gpuArray(dlX);testData.dlX=gpuArray(testData.dlX);
% csOption.plotTraining=true;
% csOption.testData=testData;
% csOption.num_partion=200;csOption.maxEpoch=10;
% csOption.h_trainingModel=@(dlX,parameters) model_CS(dlX,parameters,true);
% csOption.h_testingModel=@model_CS;
% [parameters_cs,hf1]=trainModel_CS(dlX,dlY_Target,parameters_cs,csOption);
% 
% % save ----------
% savePng(fullfile(tag_experiment,'training loss for model_CS.png'),hf1)
% for nCell=1:numel(parameters_cs);parameters_cs{nCell}=fieldfun(@gather,parameters_cs{nCell});end
% save(fullfile(paraFolder,'parameters_cs.mat'),'normalizer_CS','parameters_cs','xVarName_cs','varName2portion')

%%%*** Results using model_CS  *************************************************
disp('********************* Results using model_CS **********************')
normPara=normalizer_CS;
normalizer_CS=normClassGeneral({});normalizer_CS.varName_all=normPara.varName_all;
normalizer_CS.mean_all=normPara.mean_all;normalizer_CS.std_all=normPara.std_all;

%xVarName_r={ 'temp' 'rh' 'erd_PM25' 'PM25' 'erd_ANH4' 'erd_BC'  'erd_ANO3' 'erd_ASO4' 'erd_OC'  'erd_SOA'};
for nCell=1:numel(parameters_cs);parameters_cs{nCell}=fieldfun(@gpuArray,parameters_cs{nCell});end
calAndWriteModel_CS(PollFore,xVarName_cs,varName2portion,normalizer_CS,predictingRadius,meshObj_DSpostTrainCell,meshObj_ForeTrainCell,parameters_cs)
calAndWriteModel_CS(PollFore,xVarName_cs,varName2portion,normalizer_CS,predictingRadius,meshObj_DSpostTestCell,meshObj_ForeTestCell,parameters_cs)
%%%loss by lead for model_ST and model_CS **************************************************
disp('getERDLoss for testing data **********************')
[erdLossCell_test_ori,erdLossCell_test_st,erdLossCell_test_erd]=getERDLoss(PollFore,meshObj_ForeTestCell,meshObj_DSpostTestCell,lossRadius);
disp('getERDLoss for training data **********************')
[erdLossCell_train_ori,erdLossCell_train_st,erdLossCell_train_erd]=getERDLoss(PollFore,meshObj_ForeTrainCell,meshObj_DSpostTrainCell,lossRadius);

% loss by lead ---------------
locMean=@(x) mean(cell2mat(x),2,'omitnan');
data2plot=[locMean(erdLossCell_test_ori),locMean(erdLossCell_test_st),locMean(erdLossCell_test_erd),...
    locMean(erdLossCell_train_ori),locMean(erdLossCell_train_st),locMean(erdLossCell_train_erd)];
figure;
hl=plot(data2plot,'-*');
hl(1).Color='b';hl(2).Color='b';hl(3).Color='b';hl(4).Color='r';hl(5).Color='r';hl(6).Color='r';
hl(1).Marker='o';hl(4).Marker='o';
hl(2).Marker='+';hl(5).Marker='+';
legend('original test','eErr predicted test','erdErr predicted test','original train','eErr predicted train','erdErr predicted train')
title('loss by lead');grid on
savePng(fullfile(tag_experiment,'loss by lead for model_ST and model_CS.png'))

% loss bar
meanAllCellFun=@(x) mean(cell2mat(x),'all','omitnan');
data2plot=[meanAllCellFun(erdLossCell_train_ori),meanAllCellFun(erdLossCell_test_ori);...
    meanAllCellFun(erdLossCell_train_st),meanAllCellFun(erdLossCell_test_st);...
    meanAllCellFun(erdLossCell_train_erd),meanAllCellFun(erdLossCell_test_erd);]';
figure;
bar(data2plot)
set(gca,'XTickLabel',{'Training','Testing'})
legd=legend('original','model_ST','model_ST + model_CS','Interpreter','none','Location','southwest');
ylabel('RMSE of erdErr')

savePng(fullfile(tag_experiment,'loss of erdErr for model_ST and model_CS.png'))
%%%ForecastAndGetLoss for test data  ***********************************************************

Times2evaluate_training=[winterMonth_training:hours(1):(dateshift(winterMonth_training,'start','month','next')-hours(1)),...
    summerMonth_training:hours(1):(dateshift(summerMonth_training,'start','month','next')-hours(1))];
Times2evaluate_testing=[winterMonth_forecast:hours(1):(dateshift(winterMonth_forecast,'start','month','next')-hours(1)),...
    summerMonth_forecast:hours(1):(dateshift(summerMonth_forecast,'start','month','next')-hours(1))];
%if ~isempty(gcp('nocreate'));delete(gcp('nocreate'));end;parpool(6);% parallel ------------
disp('ForecastAndGetLoss  for testing data using model_e and model_rd **************')
[lossByLead_test_ori,lossByLead_test_ietm,lossByLead_test_st,lossByLead_test_erd]=ForecastAndGetLoss(PollFore,meshObj_ForeTestCell,meshObj_DSpostTestCell,gridWeight_predict,lossRadius,Times2evaluate_testing);

% plot result
meanFun_loc=@(x) mean(cell2mat(x),2,'omitnan');
data2plot=[meanFun_loc(lossByLead_test_ori),meanFun_loc(lossByLead_test_ietm),meanFun_loc(lossByLead_test_st),meanFun_loc(lossByLead_test_erd)];
figure;
plot(data2plot,'-*');legend('original','IETM','by model\_ST','by model\_ST  and model\_CS','Location','southeast')
title('loss for test');grid on;xlabel('lead (hour)');ylabel('RMSE');
savePng(fullfile(tag_experiment,'forecast loss by lead for test data.png'))

% do the forecasts for train data  ***********************************************************
disp('ForecastAndGetLoss for train data using model_e and model_rd **************')
[lossByLead_train_ori,lossByLead_train_ietm,lossByLead_train_e,lossByLead_train_erd]=ForecastAndGetLoss(PollFore,meshObj_ForeTrainCell,meshObj_DSpostTrainCell,gridWeight_predict,lossRadius,Times2evaluate_training);

% plot result
meanFun_loc=@(x) mean(cell2mat(x),2,'omitnan');
data2plot=[meanFun_loc(lossByLead_train_ori),meanFun_loc(lossByLead_train_ietm),meanFun_loc(lossByLead_train_e),meanFun_loc(lossByLead_train_erd)];
figure;
plot(data2plot,'-*');legend('original','IETM','by model\_ST','by model\_ST  and model\_CS','Location','southeast')
title('loss for train');grid on;xlabel('lead (hour)');ylabel('RMSE');
savePng(fullfile(tag_experiment,'forecast loss by lead for train data.png'))


%parameters_e=fieldfun(@gather,parameters_e);state_erd=fieldfun(@gather,state_erd);
reset(gpuDevice(1));if ~isempty(gcp('nocreate'));delete(gcp('nocreate'));end % parallel ------------

%% ### Training model_balance #########################################################################################################
disp('************* training the model_balance ********************')

% xVarInfo  varName_init varName_forecast
xVarName_b={ ['erdErrPredicted_' PollFore ] ['advErr_' PollFore] PollFore };

% ------------------------ prepare data ---------------------------------
folderTmp=fullfile(trainDataFolder,tag_experiment);if ~exist(folderTmp,'dir');mkdir(folderTmp);end
fileNameFun_trainingData=@(name) fullfile(folderTmp,name);
% training data --------------
fileName_dlX_bInit=fileNameFun_trainingData('train_dlX_bInit.nc');
fileName_dlX_b=fileNameFun_trainingData('train_dlX_b.nc');
fileName_bErrAllTarget=fileNameFun_trainingData('train_bErrAllTarget.nc');
fileName_bOther=fileNameFun_trainingData('train_bOther.nc');
prepareMesh_modelBalance(PollFore,xVarName_b,lossRadius,fileName_dlX_bInit,fileName_dlX_b,fileName_bErrAllTarget,fileName_bOther,meshObj_InitTrainCell,meshObj_ForeTrainCell,meshObj_DSpostTrainCell)
% testing data --------------
testFileName.fileName_dlX_bInit=fileNameFun_trainingData('test_dlX_bInit.nc');
testFileName.fileName_dlX_b=fileNameFun_trainingData('test_dlX_b.nc');
testFileName.fileName_bErrAllTarget=fileNameFun_trainingData('test_bErrAllTarget.nc');
testFileName.fileName_bOther=fileNameFun_trainingData('test_bOther.nc');
prepareMesh_modelBalance(PollFore,xVarName_b,lossRadius,testFileName.fileName_dlX_bInit,testFileName.fileName_dlX_b,testFileName.fileName_bErrAllTarget,testFileName.fileName_bOther,meshObj_InitTestCell,meshObj_ForeTestCell,meshObj_DSpostTestCell)

% init parameters ----------------------------------
[para_balance,state_balance,config]=init_netST(sizeGrid);
para_balance=fieldfun(@gpuArray,para_balance);state_balance=fieldfun(@gpuArray,state_balance);
%
for epoch=1:10
    tag_curr=['_epoch' num2str(epoch)];
    clear bOption
    % training ----------------------------------
    % eOption.testFileName=testFileName;
    bOption.epoch=1;bOption.plotTraining=true;%bOption.validationFrequency=5;bOption.testFileName=testFileName;
    para_balance=fieldfun(@gpuArray,para_balance);state_balance=fieldfun(@gpuArray,state_balance);
    [para_balance,hf1]=trainModel_balance(xVarName_b,fileName_dlX_bInit,fileName_dlX_b,fileName_bErrAllTarget,fileName_bOther,gridWeight_predict>0,para_balance,state_balance,bOption);
    % saving -------------------------------
    savePng(fullfile(tag_experiment,['training loss for model_balance' tag_curr '.png']),hf1);close(hf1);
    para_balance=fieldfun(@gather,para_balance);state_balance=fieldfun(@gather,state_balance);
    save(fullfile(paraFolder,['network_balance' tag_curr '.mat']),"para_balance");%,'state_st')
end

% % training ----------------------------------
% % eOption.testFileName=testFileName;
% bOption.epoch=20;bOption.plotTraining=true;bOption.validationFrequency=5;bOption.testFileName=testFileName;
% [para_balance,hf1]=trainModel_balance(xVarName_b,fileName_dlX_bInit,fileName_dlX_b,fileName_bErrAllTarget,fileName_bOther,gridWeight_predict>0,para_balance,state_balance,bOption);
% % saving -------------------------------
% savePng(fullfile(tag_experiment,'training loss for model_balance.png'),hf1)
% para_balance=fieldfun(@gather,para_balance);state_balance=fieldfun(@gather,state_balance);
% save(fullfile(tag_experiment,'network_balance.mat'),"para_balance",'state_balance')
%%%Results using model_balance
if ~isempty(gcp('nocreate'));delete(gcp('nocreate'));end;parpool(5);% parallel ------------
disp('********************* Results using model_balance **********************')

paraFileName=fullfile(paraFolder,'network_balance_epoch10.mat');
% prepare dlX dlY --------------------
load(paraFileName,'para_balance')
para_balance=fieldfun(@gpuArray,para_balance);state_balance=fieldfun(@gpuArray,state_balance);
%
disp('train')
calAndWriteModel_balance(meshObj_DSpostTrainCell,xVarName_b,fileName_dlX_bInit,fileName_dlX_b,fileName_bErrAllTarget,fileName_bOther,gridWeight_predict>0,para_balance,state_balance);
%
disp('test')
calAndWriteModel_balance(meshObj_DSpostTestCell,xVarName_b,testFileName.fileName_dlX_bInit,testFileName.fileName_dlX_b,testFileName.fileName_bErrAllTarget,testFileName.fileName_bOther,gridWeight_predict>0,para_balance,state_balance);

%%%getForecastLoss for test data  ***********************************************************

Times2evaluate_training=[winterMonth_training:hours(1):(dateshift(winterMonth_training,'start','month','next')-hours(1)),...
    summerMonth_training:hours(1):(dateshift(summerMonth_training,'start','month','next')-hours(1))];
Times2evaluate_testing=[winterMonth_forecast:hours(1):(dateshift(winterMonth_forecast,'start','month','next')-hours(1)),...
    summerMonth_forecast:hours(1):(dateshift(summerMonth_forecast,'start','month','next')-hours(1))];
%if ~isempty(gcp('nocreate'));delete(gcp('nocreate'));end;parpool(6);% parallel ------------
disp('ForecastAndGetLoss  for testing data using model_ST model_CS and model_Balance **************')
varForecastCell={'ConPred_ori','ConPred_erd','ConPred'};
lossCell_test=getForecastLoss(PollFore,varForecastCell,meshObj_ForeTestCell,meshObj_DSpostTestCell,lossRadius,Times2evaluate_testing);

% plot result
meanFun_loc=@(x) mean(cell2mat(x),2,'omitnan');
data2plot=[meanFun_loc(lossCell_test(1,:)),meanFun_loc(lossCell_test(2,:)),meanFun_loc(lossCell_test(3,:))];
figure;
plot(data2plot,'-*');legend('original','by model\_ST  and model\_CS','ProNet','Location','southeast')
title('loss for test');grid on;xlabel('lead (hour)');ylabel('RMSE');
savePng(fullfile(tag_experiment,'ProNet forecast loss by lead for test data.png'))

% do the forecasts for train data  ***********************************************************
disp('ForecastAndGetLoss for train data using model_e and model_rd **************')
lossCell_train=getForecastLoss(PollFore,varForecastCell,meshObj_ForeTrainCell,meshObj_DSpostTrainCell,lossRadius,Times2evaluate_training);

% plot result
meanFun_loc=@(x) mean(cell2mat(x),2,'omitnan');
data2plot=[meanFun_loc(lossCell_train(1,:)),meanFun_loc(lossCell_train(2,:)),meanFun_loc(lossCell_train(3,:))];
figure;
plot(data2plot,'-*');legend('original','by model\_ST  and model\_CS','ProNet','Location','southeast')
title('loss for train');grid on;xlabel('lead (hour)');ylabel('RMSE');
savePng(fullfile(tag_experiment,'ProNet forecast loss by lead for train data.png'))
%%%Prepare staData for the following evaluation
meshObj_DSpostFileNameAll=cellfun(@(c) c.fileName,meshObj_DSpostAllCell,'UniformOutput',false);
objCurr=meshObj_DSpostAllCell{1};  nZs=objCurr.nZs;  sizeLonLat=objCurr.sizeLonLat;leadTimes=objCurr.leadTimes;varNames=objCurr.varNames;
meshObjOut=combineMeshData(meshObj_DSpostFileNameAll,postCombineFileName,sizeLonLat,nZs,leadTimes,varNames,tStarts_all);


load staInfoAll.mat staInfo 
outStaFAdvNNFileName=fullfile(outDataFolder,['staFPost' PollFore '.mat']);
staObj_Fore=extractStaFromMeshData(postCombineFileName,lonM,latM,staInfo,[],[],{'ConPred'},tStarts_forecast);
staObj_Fore.varName={PollFore};
staObj_Fore.write2mat(outStaFAdvNNFileName);
%%%preparing for stations evalution
% read staData
figureCommonSetting_est

%dateTag='20191215';
dateTag_all={'20191215' '20200615'};% {'20161215' '20170615' '20191215' '20200615'} 
%pollName='O3';%,'PM25','PM10','SO2','NO2','CO','O3'};
for n_tag=1:numel(dateTag_all)
    dateTag=dateTag_all{n_tag};
    figureFolder=fullfile(tag_experiment,dateTag);if ~exist(figureFolder,'dir');mkdir(figureFolder);end
    tMonth=dateshift(datetime(dateTag,'InputFormat','yyyyMMdd','TimeZone','Asia/Shanghai'), 'start','month','next');
    tStarts_all=(tMonth-days(7)):days(1):(dateshift(tMonth,'start','month','next')-days(1));tStarts_all.TimeZone='Asia/Shanghai';tStarts_all.Hour=0;
    Times2evaluate_all=tMonth:hours(1):(dateshift(tMonth,'start','month','next')-hours(1));Times2evaluate_all.TimeZone='Asia/Shanghai';
    %dateTag='20200615';
    staObsFileName=fullfile(['staObs' PollFore '_' dateTag '.mat']);
    staDAFile=fullfile('D:\data_proNet_ori',['sta' PollFore 'Analysis' dateTag '.mat']);
    staFOriFile=fullfile('D:\data_proNet_ori',['staF' PollFore '_ByStart' dateTag '.mat']);
    staObs=staDATA(staObsFileName);
    Times2cal=Times2evaluate_all;
    staReanalysis=staDATA(staDAFile);
    staFOri=staDATA(staFOriFile);
    staFPost=staDATA(outStaFAdvNNFileName);
    % evaluating stations and areas

    load staInfo_verify1911.mat staInfo
    staInfo_verify=staInfo;
    yrdCities={'南京市' '临安市' '泰州市' '张家港市' '常熟市' '太仓市' '昆山市'	'南通市'	'句容市'	'吴江市'	'宜兴市'	'常州市'	'扬州市'	'无锡市'	'江阴市'	'海门市'	'淮安市'	'溧阳市'	'盐城市'	'苏州市'	'金坛市'	'镇江市'	'义乌市'	'台州市'	'嘉兴市'	'宁波市'	'富阳市'	'杭州市'	'湖州市'	'舟山市'	'金华市'	'绍兴市'	'合肥市'	'安庆市'	'宣城市'	'池州市'	'滁州市'	'芜湖市'	'蚌埠市'	'铜陵市'	'马鞍山市'	'上海市'};
    prdCities={'东莞市'	'中山市'	'云浮市'	'佛山市'	'广州市'	'惠州市'	'汕尾市'	'江门市'	'河源市'	'深圳市'	'清远市'	'珠海市'	'肇庆市'	'阳江市'	'韶关市'};
    bthProvinces={'北京' '天津' '河北'};
    twoLakeProvince={'湖南','湖北'};
    q_all=true(numel(staInfo_verify.staID),1);
    q_bth=ismember(staInfo.province,bthProvinces);
    q_yrd=ismember(staInfo.city,yrdCities);
    q_prd=ismember(staInfo.city,prdCities);
    q_twoLake=ismember(staInfo.province,twoLakeProvince);
    % preparing the staData in forcast format and matrix of dayStaLead

    % prepare observation and reanalysis in forecast format
    varS={PollFore};
    tStartsAll=tStarts_all;
    leadTimesAll=staFPost.leadTimes;
    staFObs=staObs.toForeForm(tStartsAll,varS,staInfo_verify.staID,leadTimesAll);
    staFReanalysis=staReanalysis.toForeForm(tStartsAll,varS,staInfo_verify.staID,leadTimesAll);

    % calculate the data to plot
    forecastLead=leadsAll_forecast;
    confidence_ori=nan(numel(forecastLead),2);confidence_LSTMNN=confidence_ori;
    obs_dayStaLead=nan(numel(tStartsAll),numel(staInfo_verify.staID),numel(forecastLead));
    ori_dayStaLead=obs_dayStaLead;da_dayStaLead=obs_dayStaLead;LSTMNN_dayStaLead=obs_dayStaLead;reanalysis_dayStaLead=obs_dayStaLead;
    for n_lead=1:numel(forecastLead)
        lead_curr=forecastLead(n_lead);
        t_forecasted=tStartsAll+hours(n_lead-1);
        [lia,lob]=ismember(t_forecasted,Times2cal);
        tStarts2cal=tStartsAll(lia);

        obs_dayStaLead(lia,:,n_lead)=squeeze(staFObs.get(tStarts2cal,varS,staInfo_verify.staID,lead_curr));
        ori_dayStaLead(lia,:,n_lead)=squeeze(staFOri.get(tStarts2cal,varS,staInfo_verify.staID,lead_curr));
        LSTMNN_dayStaLead(lia,:,n_lead)=squeeze(staFPost.get(tStarts2cal,varS,staInfo_verify.staID,lead_curr));    
        reanalysis_dayStaLead(lia,:,n_lead)=squeeze(staFReanalysis.get(tStarts2cal,varS,staInfo_verify.staID,lead_curr));
    end

    %% surface of RMSE
    % plot RMSE

    rmseByStaFun=@(x,y) squeeze(sqrt(mean((x-y).^2,2,'omitnan')));
    ori_rmse_dayLead=rmseByStaFun(obs_dayStaLead,ori_dayStaLead);
    lstmnn_rmse_dayLead=rmseByStaFun(obs_dayStaLead,LSTMNN_dayStaLead);
    analysis_rmse_dayLead=rmseByStaFun(obs_dayStaLead,reanalysis_dayStaLead);

    tStartAll=tStarts_all;
    hf=figure('Position',[21           85.8         1358.4          637.6]);
    data2plot_cell={ori_rmse_dayLead',lstmnn_rmse_dayLead',lstmnn_rmse_dayLead'-ori_rmse_dayLead',analysis_rmse_dayLead'};
    title2plot_cell={'RMSE of CTMf','RMSE of LSTMNN','LSTMNN-CTMf','Analysis'};
    upperLimit=prctile(ori_rmse_dayLead(:),98);
    cLimit_cell={[0 upperLimit],[0 upperLimit],[-1 1]*upperLimit*2/3,[0 upperLimit]};
    for n_plot=1:numel(data2plot_cell)
        cLimit=cLimit_cell{n_plot};
        data2plot=data2plot_cell{n_plot};
        titleName=title2plot_cell{n_plot};
        ha=subaxis(1,4,n_plot,'MB',.07,'ML',.03,'MR',.01,'MT',.045,'SH',.03);
        %[X,Y]=ndgrid(1:numel(tStartAll),leadTimesAll);
        data2plot=extendDataForPColor(data2plot);
        hp=pcolor(data2plot);
        colormap(ha,jet);colorbar(ha);caxis(ha,cLimit);
        ylabel('lead time (h)');xlabel('start day')
        ha.YTick=(1:12:numel(leadTimesAll))+0.5;
        ha.YTickLabel=cellstr(num2str(leadTimesAll(1:12:end)));
        xticks=3:5:numel(tStartAll);
        set(ha,'XTick',xticks+0.5)
        set(ha,'XTickLabel',strsplit(num2str(tStartAll.Day(xticks))));ha.TickLength=[0,0];
        title(titleName)
    end

    titleName='staLead_rmse';
    savePng(fullfile(figureFolder,titleName),hf);
    %% 
    % plot BIAS

    biasByStaFun=@(obs,pred) squeeze(mean(pred-obs,2,'omitnan'));
    ori_bias_dayLead=biasByStaFun(obs_dayStaLead,ori_dayStaLead);
    lstmnn_bias_dayLead=biasByStaFun(obs_dayStaLead,LSTMNN_dayStaLead);
    analysis_bias_dayLead=biasByStaFun(obs_dayStaLead,reanalysis_dayStaLead);

    tStartAll=staFPost.time;
    hf=figure('Position',[21           85.8         1358.4          637.6]);
    data2plot_cell={ori_bias_dayLead',lstmnn_bias_dayLead',lstmnn_bias_dayLead'-ori_bias_dayLead',analysis_bias_dayLead'};
    title2plot_cell={'BIAS of CTMf','BIAS of LSTMNN','LSTMNN-CTMf','Analysis'};
    upperLimit=prctile(abs(ori_bias_dayLead(:)),98);
    cLimit_cell={[-1 1]*upperLimit,[-1 1]*upperLimit,[-1 1]*upperLimit,[-1 1]*upperLimit};
    for n_plot=1:numel(data2plot_cell)
        cLimit=cLimit_cell{n_plot};
        data2plot=data2plot_cell{n_plot};
        titleName=title2plot_cell{n_plot};
        ha=subaxis(1,4,n_plot,'MB',.07,'ML',.03,'MR',.01,'MT',.045,'SH',.03);
        data2plot=extendDataForPColor(data2plot);
        hp=pcolor(data2plot);
        colormap jet;colorbar;caxis(cLimit);
        ylabel('lead time (h)');xlabel('start day')
        ha.YTick=(1:12:numel(leadTimesAll))+0.5;
        ha.YTickLabel=cellstr(num2str(leadTimesAll(1:12:end)));
        xticks=5:5:numel(tStartAll);
        set(gca,'XTick',xticks+0.5)
        set(gca,'XTickLabel',strsplit(num2str(tStartAll.Day(xticks))));ha.TickLength=[0,0];
        title(titleName)
    end
    titleName='staLead_bias';
    savePng(fullfile(figureFolder,titleName),hf);
    %% rmse series by lead
    nanrmseFun=@(x,y) sqrt(mean((x-y).^2,'omitnan'));
    data2plot=nan(numel(leadsAll_forecast),3);
    for n_lead=1:numel(forecastLead)
        rmseAlead=[cellfun(nanrmseFun, num2cell(obs_dayStaLead(:,:,n_lead),1)',num2cell(ori_dayStaLead(:,:,n_lead),1)'),...
            cellfun(nanrmseFun, num2cell(obs_dayStaLead(:,:,n_lead),1)',num2cell(LSTMNN_dayStaLead(:,:,n_lead),1)'),...
            cellfun(nanrmseFun, num2cell(obs_dayStaLead(:,:,n_lead),1)',num2cell(reanalysis_dayStaLead(:,:,n_lead),1)')];

        data2plot(n_lead,:)=nanmean(rmseAlead);

        confidence_ori(n_lead,:)=bootci(100,@nanmean,rmseAlead(:,1))';
        confidence_LSTMNN(n_lead,:)=bootci(100,@nanmean,rmseAlead(:,2))';
    end

    h_f=figure('units','inches','position',[3       3     figureHalfWidth  2.8]);
    %h_f=figure('Position',[1457   381   350   268]);
    h_a=axes('Position',[0.17123     0.15672     0.80515      0.8209]);
    ha=plot(forecastLead,data2plot,'LineWidth',2);

    h_conOri=patch([forecastLead(:);flipud(forecastLead(:))],[confidence_ori(:,1);flipud(confidence_ori(:,2))],[30,144,255]/255,'LineStyle','none','FaceAlpha',0.5);
    uistack(h_conOri,'bottom')
    h_conOri=patch([forecastLead(:);flipud(forecastLead(:))],[confidence_LSTMNN(:,1);flipud(confidence_LSTMNN(:,2))],[50 205 50]/255,'LineStyle','none','FaceAlpha',0.5);
    uistack(h_conOri,'bottom')

    set(gca,'XTick',24*[1 2 3 4])
    xlabel('Lead time (h)')
    ylabel('RMSE (μg m^-^3)')
    ha(3).Color=[255 69 0]/255;ha(2).Color=[64 224 208]/255;grid on
    uistack(ha(2),'bottom');
    %for toc
    box off;grid off;
    xlim([0 max(forecastLead)+1]);
    h_f.Position=[3   3   figureHalfWidth*3/4   2.8];
    h_a.Position=[0.18564     0.15077     0.78323      0.8209];
    % for main text
    box on;grid on;xlim([0 max(forecastLead)]);
    h_f.Position=[3   3   figureHalfWidth   2.8];
    h_a.Position=[0.15453     0.15104     0.83333     0.82658];
    h_leg=legend(ha,'CTMf','LSTMNN','Assimilated',...
        'Location','southeast','Orientation','horizontal');%,'Box','off'
    h_leg.Position=[0.046875      0.9208     0.94189    0.070632];
    disp(nanmean(data2plot))

    titleName='rmseLeadSeries';
    savePng(fullfile(figureFolder,titleName),h_f);
    %% statistic table

    forecastCell={ori_dayStaLead,LSTMNN_dayStaLead};
    regionCell={q_all,q_bth,q_yrd,q_prd,q_twoLake};
    dataAll=nan(numel(regionCell)*numel(forecastCell),3*4);
    for n_day=1:4
        leadtimes=(1:24)+24*(n_day-1);
        for n_region=1:numel(regionCell)
            q_region=regionCell{n_region};
            for n_forecast=1:numel(forecastCell)
                obs_currData=reshape(permute(obs_dayStaLead(:,q_region,leadtimes),[1 3 2]),[],sum(q_region));
                forecast_currData=reshape(permute(forecastCell{n_forecast}(:,q_region,leadtimes),[1 3 2]),[],sum(q_region));

                % average over stations
                tmp_obs=num2cell(obs_currData,1);
                tmp_forecast=num2cell(forecast_currData,1);
                [Rmse,Corr,Bias]=cellfun(@ calStatistic,tmp_obs,tmp_forecast);
                % bias
                dataAll(n_forecast+((n_region-1)*numel(forecastCell)), 1+3*(n_day-1))=nanmean(Bias);
                % rmse
                dataAll(n_forecast+((n_region-1)*numel(forecastCell)), 2+3*(n_day-1))=nanmean(Rmse);
                % corr
                dataAll(n_forecast+((n_region-1)*numel(forecastCell)), 3+3*(n_day-1))=nanmean(Corr);
            end
        end
    end
    columnNameS={'1dayF_bias','1dayF_rmse','1dayF_corr',...
        '2dayF_bias','2dayF_rmse','2dayF_corr',...
        '3dayF_bias','3dayF_rmse','3dayF_corr',...
        '4dayF_bias','4dayF_rmse','4dayF_corr'};
    rowNameS={'ori_all','ori_bth','ori_yrd','ori_prd','ori_twoLake';...
        'corrected_all','corrected_bth','corrected_yrd','corrected_prd','corrected_twoLake'};
    rowNameS=rowNameS(:);
    statisticT=array2table(dataAll,'RowNames',rowNameS,'VariableNames' ,columnNameS)
    writetable(statisticT,fullfile(figureFolder,'area statistic.xlsx'),'WriteRowNames' ,true)
    %% 
    % plot RMSE

    forecastCell={ori_dayStaLead,LSTMNN_dayStaLead};
    RMSECell=cell(size(forecastCell));
    for n_day=1:4
        leadtimes=(1:24)+24*(n_day-1);
        q_region=q_all;
        for n_forecast=1:numel(forecastCell)
            obs_currData=reshape(permute(obs_dayStaLead(:,q_region,leadtimes),[1 3 2]),[],sum(q_region));
            forecast_currData=reshape(permute(forecastCell{n_forecast}(:,q_region,leadtimes),[1 3 2]),[],sum(q_region));

            % average over stations
            tmp_obs=num2cell(obs_currData,1);
            tmp_forecast=num2cell(forecast_currData,1);
            [Rmse,Corr,Bias]=cellfun(@ calStatistic,tmp_obs,tmp_forecast);
            % rmse
            RMSECell{n_forecast}=Rmse;
        end

        load lonMlatM_d2.mat lonM latM
        %data2plot=[statistic_staForecast.rmse,statistic_staDAForecast.rmse,statistic_staPostForecast.rmse,statistic_staPostForecast.rmse-statistic_staDAForecast.rmse];
        data2plot=[RMSECell{1}',RMSECell{2}',RMSECell{2}'-RMSECell{1}'];
        %h_f=figure('Position',[66.2           69.8          687.2          616.8]);
        if 1==n_day
            cLimit_con=[0 prctile(data2plot(:,1),95)];
            cLimit_change=[-1 1]*prctile(abs(data2plot(:,3)),95);
        end
        h_f=figure('units','inches','position',[1 1     figureFullWidth  2]);
        for n_sub=1:3
            hax=subaxis(1,3,n_sub,'ML',.06,'MR',.08,'MT',.001,'MB',.035,'SV',.05,'SH',.13); 
            ham=axesm('lambert','mapparallels',[4 44],'origin',[0 106 0]);
            h_s=scatterm(staInfo_verify.lat,staInfo_verify.lon,20,data2plot(:,n_sub),'filled');
            colormap jet;h_bar=colorbar;
            h_bar.Position(1)=hax.Position(1)+hax.Position(3)+0.017;
            h_bar.Position(3)=0.017984;    
            title(h_bar,' μg m^{-3}')
            if 3==n_sub
                caxis(cLimit_change)
            else
                caxis(cLimit_con)
            end
            h_ss2=plotMap_d2(h_f,hax,lonM,latM);
            h_grid=findobj(gca,'LineStyle',':');
            for n=1:numel(h_grid);h_grid(n).Color=[1 1 1]*0.4;end      
            h_ss2.Position(1)=h_ss2.Position(1)-(0.33375-0.33229);  
            h_ss2.Position(2)=h_ss2.Position(2)-(0.542-0.53714);
        end
        % 'CTM with non-assimilated ICs','Traditional method','New method'
        annotation('textbox',[0.056648     0.86836     0.13239    0.045314],'String','(a) CTMf','BackgroundColor','w','EdgeColor','none','Margin',2);
        annotation('textbox',[0.39355     0.86836     0.13239    0.045314],'String','(c) LSTMNN','BackgroundColor','w','EdgeColor','none','Margin',2);
        annotation('textbox',[0.71924     0.86836     0.13239    0.045314],'String','(d) LSTMNN - CTMf','BackgroundColor','w','EdgeColor','none','Margin',2);

        titleName=[num2str(n_day) 'day scatter RMSE'];
        savePng(fullfile(figureFolder,titleName),h_f);
    end
    %% 
    % plot Bias

    forecastCell={ori_dayStaLead,LSTMNN_dayStaLead};
    BIASCell=cell(size(forecastCell));
    for n_day=1:4
        leadtimes=(1:24)+24*(n_day-1);
        q_region=q_all;
        for n_forecast=1:numel(forecastCell)
            obs_currData=reshape(permute(obs_dayStaLead(:,q_region,leadtimes),[1 3 2]),[],sum(q_region));
            forecast_currData=reshape(permute(forecastCell{n_forecast}(:,q_region,leadtimes),[1 3 2]),[],sum(q_region));

            % average over stations
            tmp_obs=num2cell(obs_currData,1);
            tmp_forecast=num2cell(forecast_currData,1);
            [Rmse,Corr,Bias]=cellfun(@ calStatistic,tmp_obs,tmp_forecast);
            % rmse
            BIASCell{n_forecast}=Bias;
        end

        load lonMlatM_d2.mat lonM latM
        %data2plot=[statistic_staForecast.rmse,statistic_staDAForecast.rmse,statistic_staPostForecast.rmse,statistic_staPostForecast.rmse-statistic_staDAForecast.rmse];
        data2plot=[BIASCell{1}',BIASCell{2}',BIASCell{2}'-BIASCell{1}'];
        %h_f=figure('Position',[66.2           69.8          687.2          616.8]);

        if 1==n_day
            cLimit_bias=[-1 1]*prctile(abs(data2plot(:,1)),95);
        end
        h_f=figure('units','inches','position',[1 1     figureFullWidth  2]);
        for n_sub=1:3
            hax=subaxis(1,3,n_sub,'ML',.06,'MR',.08,'MT',.001,'MB',.035,'SV',.05,'SH',.13); 
            ham=axesm('lambert','mapparallels',[4 44],'origin',[0 106 0]);
            h_s=scatterm(staInfo_verify.lat,staInfo_verify.lon,20,data2plot(:,n_sub),'filled');
            colormap jet;h_bar=colorbar;
            h_bar.Position(1)=hax.Position(1)+hax.Position(3)+0.017;
            h_bar.Position(3)=0.017984;    
            title(h_bar,' μg m^{-3}')
            if 3==n_sub
                caxis(cLimit_bias)
            else
                caxis(cLimit_bias)
            end
            h_ss2=plotMap_d2(h_f,hax,lonM,latM);
            h_grid=findobj(gca,'LineStyle',':');
            for n=1:numel(h_grid);h_grid(n).Color=[1 1 1]*0.4;end      
            h_ss2.Position(1)=h_ss2.Position(1)-(0.33375-0.33229);  
            h_ss2.Position(2)=h_ss2.Position(2)-(0.542-0.53714);
        end
        % 'CTM with non-assimilated ICs','Traditional method','New method'
        annotation('textbox',[0.056648     0.86836     0.13239    0.045314],'String','(a) CTMf','BackgroundColor','w','EdgeColor','none','Margin',2);
        annotation('textbox',[0.39355     0.86836     0.13239    0.045314],'String','(c) LSTMNN','BackgroundColor','w','EdgeColor','none','Margin',2);
        annotation('textbox',[0.71924     0.86836     0.13239    0.045314],'String','(d) LSTMNN - CTMf','BackgroundColor','w','EdgeColor','none','Margin',2);


        titleName=[num2str(n_day) 'day scatter BIAS'];
        savePng(fullfile(figureFolder,titleName),h_f);
    end
    %% 
    % plot sample improve in stations
    tStarts2plot=tStartsAll([5 12 18]+1);
    staIDs=[110000246;340700436;441900401];
    staInfo2plot=staObs.getStaInfo(staIDs);
    leadHours=leadsAll_forecast;
    %figure('Position',[1060   792   635   214])
    h_f=figure('units','inches','position',[1 1     figureFullWidth  2.2]);
    for n_start=1:numel(tStarts2plot)
        tStart=tStarts2plot(n_start);
        tStartInModel=tStart;tStartInModel.Hour=startHour;
        if tStartInModel>tStart;tStartInModel=tStartInModel-days(1);end

        Times2plot=tStart+hours(leadHours);
        staID=staIDs(n_start);
        % get data
        dataAstart_LSTMNN=squeeze(staFPost.get(tStart,PollFore,staID,leadHours));
        dataAstart_oriF=squeeze(staFOri.get(tStartInModel,PollFore,staID,leadHours));
        data_obs=squeeze(staObs.get(Times2plot,PollFore,staID));

        dataForecast=[dataAstart_oriF,dataAstart_LSTMNN];

        subaxis(1,3,n_start,'ML',0.075,'MR',0.01,'MT',0.14,'SH',0.08);
        ha=plot(Times2plot,[dataForecast,data_obs],'LineWidth',2);
        ha(3).Color=[255 69 0]/255;ha(2).Color=[64 224 208]/255;
        uistack(ha(1),'top')
        grid on;xlim(Times2plot([1 end]));
        addPollYlabel(PollFore);
        datetick('x','mm/dd','keeplimits','keepticks');
    end
    legend(ha,'CTMf','LSTMNN','Observation','Orientation','horizontal','Location','southoutside','Position',[0.24836     0.89251      0.5619    0.098485])
    annotation('textbox',[0.072048     0.76636     0.2      0.1185],'String','(a) Beijing BTH','BackgroundColor','none','EdgeColor','none');
    annotation('textbox',[0.40276     0.76636     0.2      0.1185],'String','(b) Tongling YRD','BackgroundColor','none','EdgeColor','none');
    annotation('textbox',[0.73504     0.76636     0.2      0.1185],'String','(c) Dongguan PRD','BackgroundColor','none','EdgeColor','none');

    titleName='forecast sample';
    savePng(fullfile(figureFolder,titleName),h_f);
    toc(timerTotal)
end
%% functions for model_ST 
function prepareMesh_modelST(PollFore,xVarName_e,lossRadius,fileName_dlX_eInit,fileName_dlX_e,fileName_dlY_eTarget,fileName_eLossWeight,tStarts_all,meshObj_InitTrainCell,meshObj_ForeTrainCell)

    objCurr=meshObj_InitTrainCell{1};  nZs=objCurr.nZs;  sizeLonLat=objCurr.sizeLonLat;leadsAll_init=objCurr.leadTimes(2:end);
    objCurr=meshObj_ForeTrainCell{1};  leadsAll_forecast=objCurr.leadTimes;
    customOrder=meshObj_ForeTrainCell{1}.customDimOrder;
    varTarget=['erd_' PollFore 'Err'];

    mesh_dlX_eInit=meshDATA(fileName_dlX_eInit,sizeLonLat,nZs,leadsAll_init,xVarName_e,tStarts_all);mesh_dlX_eInit.setCustomDimOrder(customOrder);
    mesh_dlX_e=meshDATA(fileName_dlX_e,sizeLonLat,nZs,leadsAll_forecast,xVarName_e,tStarts_all);mesh_dlX_e.setCustomDimOrder(customOrder);
    mesh_dlY_eTarget=meshDATA(fileName_dlY_eTarget,sizeLonLat,nZs,leadsAll_forecast,{varTarget},tStarts_all);mesh_dlY_eTarget.setCustomDimOrder(customOrder);
    mesh_eLossWeight=meshDATA(fileName_eLossWeight,sizeLonLat,nZs,leadsAll_forecast,{'lossWeight'},tStarts_all);mesh_eLossWeight.setCustomDimOrder(customOrder);
    
WaitMessage = parfor_wait(numel(tStarts_all),'Waitbar',true,'Title','prepareMesh_modelST');
    for n_start=1:numel(tStarts_all)
        tStart_curr=tStarts_all(n_start);
        q_before=contains(xVarName_e,'Before');
        meshIn_init=meshObj_InitTrainCell{n_start};
        meshIn_fore=meshObj_ForeTrainCell{n_start};
        % dlX_eInit
        dlX_eInit=nan([sizeLonLat,numel(xVarName_e),1,numel(leadsAll_init)]);
        dlX_eInit(:,:,~q_before,:,:)=meshIn_init.get(nZs,leadsAll_init,xVarName_e(~q_before),tStart_curr);
        dlX_eInit(:,:,q_before,:,:)=meshIn_init.get(nZs,leadsAll_init-1,{varTarget},tStart_curr);    
        mesh_dlX_eInit.put(dlX_eInit,nZs,leadsAll_init,xVarName_e,tStart_curr);
        % dlX_e
        dlX_e=nan([sizeLonLat,numel(xVarName_e),1,numel(leadsAll_forecast)]);
        dlX_e(:,:,~q_before,:,:)=meshIn_fore.get(nZs,leadsAll_forecast,xVarName_e(~q_before),tStart_curr);
        dlX_e(:,:,q_before,:,1:2)=meshIn_init.get(nZs,[-1 0],{varTarget},tStart_curr);    
        mesh_dlX_e.put(dlX_e,nZs,leadsAll_forecast,xVarName_e,tStart_curr);
        % dlY_eTarget
        dlY_eTarget=meshIn_fore.get(nZs,leadsAll_forecast,{varTarget},tStart_curr);
        mesh_dlY_eTarget.put(dlY_eTarget,nZs,leadsAll_forecast,{varTarget},tStart_curr);
        % lossWeight
        lossWeight=single(meshIn_fore.get(nZs,leadsAll_forecast,{[PollFore 'minValidDist']},tStart_curr)<lossRadius);
        mesh_eLossWeight.put(lossWeight,nZs,leadsAll_forecast,{'lossWeight'},tStart_curr);
WaitMessage.Send;
    end
WaitMessage.Destroy;
end

function calAndWriteModel_st(meshObj_DSpostCell,xVarName_st,fileName_dlX_stInit,fileName_dlX_st,parameters_st,state_st)
    customOrder=[1 2 5 6 4 3];% [lon lat var start lead]
    mesh_dlX_stInit=meshDATA(fileName_dlX_stInit);mesh_dlX_stInit.setCustomDimOrder(customOrder);
    mesh_dlX_st=meshDATA(fileName_dlX_st);mesh_dlX_st.setCustomDimOrder(customOrder);
    tStarts_all=mesh_dlX_st.tStarts;    
WaitMessage = parfor_wait(numel(tStarts_all),'Waitbar',true,'Title','Get stErrPred');
    for nt_start=1:numel(tStarts_all)
        tStart_curr=tStarts_all(nt_start);
        dlXtest_init=mesh_dlX_stInit.get([],[],[],tStart_curr);
        dlXtest=mesh_dlX_st.get([],[],[],tStart_curr);
        dlY_Predicted = modelAstart_ST(xVarName_st,dlXtest_init,dlXtest,parameters_st,state_st);
WaitMessage.Send;
        meshObj_DSpostCell{nt_start}.put(dlY_Predicted,[],[],{'stErrPred'},tStart_curr);
    end
WaitMessage.Destroy;
end
%% functions for model_CS
function [dlXtrain_cs,dlYtrain_cs]=prepareData_modelCS(PollFore,xVarName_cs,leads4modelR,lossRadius,meshInCell,meshPostCell,Times2extract)
    tStarts_all=cellfun(@(obj) obj.tStarts,meshInCell);
    dlXCell=cell(size(tStarts_all));  dlYtrainCell_Target=cell(size(tStarts_all));
    varTarget=['erd_' PollFore 'Err'];
WaitMessage = parfor_wait(numel(tStarts_all),'Waitbar',true,'Title','prepareData_modelCS');
    for n_start=1:numel(tStarts_all)
        tStart_curr=tStarts_all(n_start);
        TimesInFile_candidate=tStart_curr+hours(leads4modelR);
        qlead2read=ismember(TimesInFile_candidate,Times2extract);
        leads2read=leads4modelR(qlead2read);        
        if isempty(leads2read);dlXCell{n_start}=nan(numel(xVarName_cs),0,'single');dlYtrainCell_Target{n_start}=nan(1,0,'single');continue;end
        q_picked=meshInCell{n_start}.get([],(min(leads2read)-1):max(leads2read),{[PollFore 'minValidDist']},tStart_curr)<lossRadius;
        q_picked=q_picked(:,:,:,:,1:end-1) & q_picked(:,:,:,:,2:end);
        % dlX_e 
        dlX_meshTmp=meshInCell{n_start}.get([],leads2read,xVarName_cs,tStart_curr);
        dlX_tmp=nan(numel(xVarName_cs),sum(q_picked,'all'));
        for n_var=1:numel(xVarName_cs)
            tmp=dlX_meshTmp(:,:,n_var,:,:);
            dlX_tmp(n_var,:)=tmp(q_picked);
        end
        dlXCell{n_start}=single(dlX_tmp);
        % dlY_eTarget
        erdErrTarget=meshInCell{n_start}.get([],leads2read,{varTarget},tStart_curr);
        stPredicted=meshPostCell{n_start}.get([],leads2read,{'stErrPred'},tStart_curr);
        dlYtrainCell_Target{n_start}=erdErrTarget(q_picked)'-stPredicted(q_picked)';
        %dlYtrainCell_Target{n_start}=erdErrTarget(q_picked)';
WaitMessage.Send;
    end
    dlXtrain_cs=cell2mat(dlXCell);dlYtrain_cs=cell2mat(dlYtrainCell_Target);
WaitMessage.Destroy;
end

function data=portionX(data,varNameIn,varName2portion,dataDenominator)
    q2portion=ismember(varNameIn,varName2portion);
    data(q2portion,:)=data(q2portion,:)./repmat(dataDenominator(:)',sum(q2portion),1,1);
end

function importanceData=getImportance_ref(dlX_r,h_model)

    sampleSize=300;
    refX=median(dlX_r,2);
    %offset=-std(dlX_r')'*0.01;
    % do the neural network part ----------------------------------
    importanceData=nan(size(dlX_r));
    WaitMessage = parfor_wait(size(dlX_r,2),'Waitbar',true,'Title','Get Shapley value');
    parfor n_grid=1:size(dlX_r,2)
        %refX=dlX_r(:,n_grid)+offset;
        y=Shaply(h_model,dlX_r(:,n_grid),sampleSize,refX);
        importanceData(:,n_grid)=y;%gather(extractdata(y));
        WaitMessage.Send;
    end
    WaitMessage.Destroy;
end

function calAndWriteModel_CS(PollFore,xVarName_r,varName2portion,normalizer_R,predictRadius,meshObj_DSpostCell,meshInCell,parameters_r)
num_partion=50;
WaitMessage = parfor_wait(numel(meshObj_DSpostCell),'Waitbar',true,'Title','calAndWriteModel_CS');
    for n_start=1:numel(meshObj_DSpostCell)
        tStart_curr=meshObj_DSpostCell{n_start}.tStarts;
        q_predict=meshInCell{n_start}.get([],[],{[PollFore 'minValidDist']},tStart_curr)<predictRadius;
        % dlX_e 
        dlX_mesh=meshInCell{n_start}.get([],[],xVarName_r,tStart_curr);
        dlY_mesh=zeros(size(dlX_mesh(:,:,1,:,:)),'single');
        dlX=zeros(numel(xVarName_r),sum(q_predict,'all'),'single');
        for n_var=1:numel(xVarName_r)
            tmp=dlX_mesh(:,:,n_var,:,:);
            dlX(n_var,:)=tmp(q_predict);
        end
        dataDenominator_mesh=meshInCell{n_start}.get([],[],{PollFore},tStart_curr);
        dataDenominator=dataDenominator_mesh(q_predict);
        dlX=portionX(dlX,xVarName_r,varName2portion,dataDenominator);
        dlX=normalizer_R.normData(dlX,xVarName_r);
        dlY=getMassiveModelRespond(dlarray(dlX),@model_CS,parameters_r,num_partion);
%         dlX=gpuArray(dlarray(dlX));
%         dlY=model_CS(dlX,parameters_r);
        dlY_mesh(q_predict)=dlY;
        meshObj_DSpostCell{n_start}.put(dlY_mesh,[],[],{'csErrPred'},tStart_curr);
WaitMessage.Send;
    end
WaitMessage.Destroy;
end

function allPredicetion=getMassiveModelRespond(dlX,h_model,parameters,num_partion)
    % ######################################## Loop over epochs ##########################################
    if size(dlX,2)>50000000;dispWaitBar=true;else;dispWaitBar=false;end
    if dispWaitBar;        WaitMessage = parfor_wait(num_partion,'Waitbar',true,'Title','getMassiveModelRespond');end
    XCell=dataPartion(num_partion,dlX);
    outCell=cell(size(XCell));
    for n_cell =1:num_partion
        dlX_curr=XCell{n_cell};
        dlX_curr=gpuArray(dlX_curr);
        outCell{n_cell}=gather(extractdata(h_model(dlX_curr,parameters)));
    if dispWaitBar;        WaitMessage.Send;end
    end
    if dispWaitBar;        WaitMessage.Destroy;end
    allPredicetion=cell2mat(outCell);
    function XCell=dataPartion(num_partion,Xin)
        num_sample=size(Xin,2);
        XCell=cell(1,num_partion);
        xbars=linspace(0,num_sample,num_partion+1);
        indSamle=1:num_sample;
        for n_partion=1:num_partion
            q_curr=indSamle > xbars(n_partion) & indSamle <= xbars(n_partion+1);
            XCell{n_partion}=Xin(:,q_curr);
        end
    end
end

function plotImportance(importanceData,xVarName_cs,dlX_cs)
    [~,ind_desc]=sort(mean(abs(importanceData),2),'desc');
    ha=axes('Position',[0.04294     0.19524     0.91164     0.74286]);
    data2plot=mat2cell(importanceData(:,:)/50,ones(1,size(importanceData,1)),size(importanceData,2));
    tmp=dlX_cs';tmp(tmp-min(tmp)==0)=nan;
    colorRefs=prctile(tmp,[5 50 95])';
    colorData=mat2cell((dlX_cs-colorRefs(:,2))./(min(diff(colorRefs,1,2),[],2)),ones(1,size(dlX_cs,1)),size(dlX_cs,2));
    plotSpread(data2plot(ind_desc),'xNames',xVarName_cs(ind_desc),'colorData',colorData(ind_desc))
    ha.XAxis.TickLabelInterpreter='none';
    grid on;ylabel('Importance')
    caxis([-1 1]);ylim([-1 1]*2*prctile(abs(importanceData(:)),50))
    cMap=jet(64);cMap=cMap(6:58,:);
    colormap(cMap);hbar=colorbar;hbar
end

%% functions for model_balance 
function prepareMesh_modelBalance(PollFore,xVarName_b,lossRadius,fileName_dlX_bInit,fileName_dlX_b,fileName_bErrAllTarget,fileName_bOther,...
    meshObj_InitCell,meshObj_ForeCell,meshObj_DSpostCell)
    q_erdErrPredict=contains(xVarName_b,'erdErrPredicted');
    q_advErr=contains(xVarName_b,'advErr');
    q_direct=~(q_erdErrPredict|q_advErr);
% xVarName_b={ ['erdErrPredicted_' PollFore ] ['advErr_' PollFore] PollFore };
    objCurr=meshObj_InitCell{1};  nZs=objCurr.nZs;  sizeLonLat=objCurr.sizeLonLat;leadsAll_init=objCurr.leadTimes(2:end);
    objCurr=meshObj_ForeCell{1};  leadsAll_forecast=objCurr.leadTimes;
    customOrder=meshObj_ForeCell{1}.customDimOrder;
    varName_erdErr=['erd_' PollFore 'Err'];
    varName_advErr=['adv_' PollFore 'Err'];
    varName_ErrAllTarget=['ErrAllTarget_' PollFore];
    varS_other={'lossWeight','u','v','con_ori'};
    
    tStarts_all=cellfun(@(obj) obj.tStarts,meshObj_ForeCell);

    mesh_dlX_bInit=meshDATA(fileName_dlX_bInit,sizeLonLat,nZs,leadsAll_init,xVarName_b,tStarts_all);mesh_dlX_bInit.setCustomDimOrder(customOrder);
    mesh_dlX_b=meshDATA(fileName_dlX_b,sizeLonLat,nZs,leadsAll_forecast,xVarName_b,tStarts_all);mesh_dlX_b.setCustomDimOrder(customOrder);
    mesh_dlY_bErrAllTarget=meshDATA(fileName_bErrAllTarget,sizeLonLat,nZs,leadsAll_forecast,{varName_ErrAllTarget},tStarts_all);mesh_dlY_bErrAllTarget.setCustomDimOrder(customOrder);
    mesh_bOther=meshDATA(fileName_bOther,sizeLonLat,nZs,leadsAll_forecast,{'lossWeight','u','v','con_ori'},tStarts_all);mesh_bOther.setCustomDimOrder(customOrder);
    for n_start=1:numel(tStarts_all)
        tStart_curr=tStarts_all(n_start);
        meshIn_init=meshObj_InitCell{n_start};
        meshIn_fore=meshObj_ForeCell{n_start};
        meshIn_post=meshObj_DSpostCell{n_start};
        % dlX_bInit
        dlX_bInit=nan([sizeLonLat,numel(xVarName_b),1,numel(leadsAll_init)],'single');
        dlX_bInit(:,:,q_erdErrPredict,:,:)=meshIn_init.get(nZs,leadsAll_init,{varName_erdErr},tStart_curr);
        dlX_bInit(:,:,q_advErr,:,:)=meshIn_init.get(nZs,leadsAll_init,{varName_advErr},tStart_curr);
        dlX_bInit(:,:,q_direct,:,:)=meshIn_init.get(nZs,leadsAll_init,xVarName_b(q_direct),tStart_curr);    
        mesh_dlX_bInit.put(dlX_bInit,nZs,leadsAll_init,xVarName_b,tStart_curr);
        % dlX_b
        dlX_b=nan([sizeLonLat,numel(xVarName_b),1,numel(leadsAll_forecast)],'single');
        dlX_b(:,:,q_direct,:,:)=meshIn_fore.get(nZs,leadsAll_forecast,xVarName_b(q_direct),tStart_curr);
        dlX_b(:,:,q_erdErrPredict,:,:)=sum(meshIn_post.get(nZs,leadsAll_forecast,{'stErrPred' 'csErrPred'},tStart_curr),3);
        mesh_dlX_b.put(dlX_b,nZs,leadsAll_forecast,xVarName_b,tStart_curr);
        % bErrAllTarget
        bErrAllTarget=meshIn_fore.get(nZs,leadsAll_forecast,{[PollFore 'Err']},tStart_curr);
        mesh_dlY_bErrAllTarget.put(bErrAllTarget,nZs,leadsAll_forecast,{varName_ErrAllTarget},tStart_curr);
        % bOther
        data_bOther=nan([sizeLonLat,numel(varS_other),1,numel(leadsAll_forecast)],'single');
        data_bOther(:,:,strcmp(varS_other,'u'),:,:)=meshIn_fore.get(nZs,leadsAll_forecast,{'u'},tStart_curr);
        data_bOther(:,:,strcmp(varS_other,'v'),:,:)=meshIn_fore.get(nZs,leadsAll_forecast,{'v'},tStart_curr);
        data_bOther(:,:,strcmp(varS_other,'con_ori'),:,:)=meshIn_fore.get(nZs,leadsAll_forecast,{PollFore},tStart_curr);
        lossWeight=meshIn_fore.get(nZs,leadsAll_forecast,{[PollFore 'minValidDist']},tStart_curr)<lossRadius;
        lossWeight(:,:,:,:,2:end)=lossWeight(:,:,:,:,1:end-1) & lossWeight(:,:,:,:,2:end);
        data_bOther(:,:,strcmp(varS_other,'lossWeight'),:,:)=single(lossWeight);
        mesh_bOther.put(data_bOther,nZs,leadsAll_forecast,varS_other,tStart_curr);
    end
end

function calAndWriteModel_balance(meshObj_DSpostCell,xVarName,fileName_dlX_bInit,fileName_dlX_b,fileName_bErrAllTarget,fileName_bOther,q_grid2predict,para_b,state_b)
    customOrder=[1 2 5 6 4 3];% [lon lat var start lead]
    meshObj_dlX_eInit=meshDATA(fileName_dlX_bInit);meshObj_dlX_eInit.setCustomDimOrder(customOrder);
    meshObj_dlX_e=meshDATA(fileName_dlX_b);meshObj_dlX_e.setCustomDimOrder(customOrder);
    meshObj_ErrAll_Target=meshDATA(fileName_bErrAllTarget);meshObj_ErrAll_Target.setCustomDimOrder(customOrder);
    meshObj_bOther=meshDATA(fileName_bOther);meshObj_bOther.setCustomDimOrder(customOrder);
    tStarts_all=meshObj_ErrAll_Target.tStarts;
WaitMessage = parfor_wait(numel(tStarts_all),'Waitbar',true,'Title','calAndWriteModel_balance');
    parfor nt_start=1:numel(tStarts_all)
        tStart_curr=tStarts_all(nt_start);
        % prepare data
        dlX_init=meshObj_dlX_eInit.get([],[],[],tStart_curr);
        dlX_fore=meshObj_dlX_e.get([],[],[],tStart_curr);
        ErrAll_init=meshObj_ErrAll_Target.get([],0,[],tStart_curr);
        u_aStart=meshObj_bOther.get([],[],'u',tStart_curr);
        v_aStart=meshObj_bOther.get([],[],'v',tStart_curr);
        con_ori_aStart=meshObj_bOther.get([],[],'con_ori',tStart_curr);
        [bErr_Predicted,ErrAll_Predicted] = modelAstart_Balance(xVarName,dlX_init,dlX_fore,ErrAll_init,u_aStart,v_aStart,con_ori_aStart,q_grid2predict,para_b,state_b);
        meshObj_DSpostCell{nt_start}.put(bErr_Predicted,[],[],{'bErrPred'},tStart_curr);
        meshObj_DSpostCell{nt_start}.put(ErrAll_Predicted,[],[],{'errAllPred'},tStart_curr);
        meshObj_DSpostCell{nt_start}.put(con_ori_aStart-ErrAll_Predicted,[],[],{'ConPred'},tStart_curr);
WaitMessage.Send;
    end
WaitMessage.Destroy;
end
%% functions for total results
function [erdLossCell_ori,erdLossCell_e,erdLossCell_erd]=getERDLoss(PollFore,meshObj_ForeCell,meshObj_DSpostCell,lossRadius)
    erdLossCell_ori=cell(1,numel(meshObj_DSpostCell)); 
    erdLossCell_e=cell(1,numel(meshObj_DSpostCell)); 
    erdLossCell_erd=cell(1,numel(meshObj_DSpostCell));
    WaitMessage = parfor_wait(numel(meshObj_DSpostCell),'Waitbar',true,'Title','getERDLoss'); 
    for nt_start=1:numel(meshObj_DSpostCell)
        tStart=meshObj_DSpostCell{nt_start}.tStarts;
        erdTarget_aStart=meshObj_ForeCell{nt_start}.get([],[],{['erd_' PollFore 'Err']},tStart);
        ePredict_aStart=meshObj_DSpostCell{nt_start}.get([],[],{'stErrPred'},tStart);
        erdPredict_aStart=ePredict_aStart+meshObj_DSpostCell{nt_start}.get([],[],{'csErrPred'},tStart);

        lossWeight=meshObj_ForeCell{nt_start}.get([],[],{[PollFore 'minValidDist']},tStart);
        lossWeight=lossWeight<lossRadius;    lossWeight(:,:,:,:,2:end)=lossWeight(:,:,:,:,1:(end-1)) & lossWeight(:,:,:,:,2:end);    
        erdLossCell_ori{nt_start}=calLossSeries(erdPredict_aStart.*0,erdTarget_aStart,lossWeight);    
        erdLossCell_e{nt_start}=calLossSeries(ePredict_aStart,erdTarget_aStart,lossWeight);    
        erdLossCell_erd{nt_start}=calLossSeries(erdPredict_aStart,erdTarget_aStart,lossWeight);    
    WaitMessage.Send;
    end
    WaitMessage.Destroy;
end
function lossCell=getForecastLoss(PollFore,varForecastCell,meshObj_ForeCell,meshObj_DSpostCell,lossRadius,Times2evaluate)
    lossCell=cell(numel(varForecastCell),numel(meshObj_DSpostCell)); 
    WaitMessage = parfor_wait(numel(meshObj_DSpostCell),'Waitbar',true,'Title','getERDLoss'); 
    for nt_start=1:numel(meshObj_DSpostCell)
        tStart=meshObj_DSpostCell{nt_start}.tStarts;
        Times_curr=tStart+hours(meshObj_DSpostCell{nt_start}.leadTimes);
        qt2evalute=ismember(Times_curr,Times2evaluate);
        if sum(qt2evalute)==0
            for n_fore=1:numel(varForecastCell);lossCell{n_fore,nt_start}=nan(numel(Times_curr),1);end
        else
            leads2read=meshObj_DSpostCell{nt_start}.leadTimes(qt2evalute);
            analysis_aStart=meshObj_ForeCell{nt_start}.get([],leads2read,{[PollFore 'Analysis']},tStart);
            lossWeight=meshObj_ForeCell{nt_start}.get([],leads2read,{[PollFore 'minValidDist']},tStart);
            lossWeight=lossWeight<lossRadius;    lossWeight(:,:,:,:,2:end)=lossWeight(:,:,:,:,1:(end-1)) & lossWeight(:,:,:,:,2:end); 
            for n_fore=1:numel(varForecastCell)
                varForecast_curr=varForecastCell{n_fore};
                if strcmp(varForecast_curr,'ConPred_ori')
                    forecast_aStart=meshObj_ForeCell{nt_start}.get([],leads2read,{PollFore},tStart);
                else
                    forecast_aStart=meshObj_DSpostCell{nt_start}.get([],leads2read,{varForecast_curr},tStart);
                end
                loss_curr=nan(numel(Times_curr),1);loss_curr(qt2evalute)=calLossSeries(forecast_aStart,analysis_aStart,lossWeight);    
                lossCell{n_fore,nt_start}=loss_curr;
            end
        end
    WaitMessage.Send;
    end
    WaitMessage.Destroy;
end

function [lossByLead_ori,lossByLead_ietm,lossByLead_st,lossByLead_erd]=ForecastAndGetLoss(PollFore,meshObj_ForeCell,meshObj_DSpostCell,gridWeight_predict,lossRadius,Times2evaluate)
    lossByLead_ori=cell(1,numel(meshObj_DSpostCell)); 
    lossByLead_ietm=cell(1,numel(meshObj_DSpostCell)); 
    lossByLead_st=cell(1,numel(meshObj_DSpostCell));
    lossByLead_erd=cell(1,numel(meshObj_DSpostCell));
    WaitMessage = parfor_wait(numel(meshObj_DSpostCell),'Waitbar',true,'Title',': Forecasts with eErrPred and rdErrPred');
    parfor nt_start=1:numel(meshObj_DSpostCell)
        tStart=meshObj_DSpostCell{nt_start}.tStarts;
        Times_curr=tStart+hours(meshObj_DSpostCell{nt_start}.leadTimes);
        qt2evalute=ismember(Times_curr,Times2evaluate);
        % prepare data
        u_aStart=meshObj_ForeCell{nt_start}.get([],[],{'u'},tStart);
        v_aStart=meshObj_ForeCell{nt_start}.get([],[],{'v'},tStart);
        Poll_aStart=meshObj_ForeCell{nt_start}.get([],[],{PollFore},tStart);
        analysis_aStart=meshObj_ForeCell{nt_start}.get([],[],{[PollFore 'Analysis']},tStart);
        lossWeight=meshObj_ForeCell{nt_start}.get([],[],{[PollFore 'minValidDist']},tStart);
        lossWeight=double(lossWeight<lossRadius);   % lossWeight(:,:,:,:,2:end)=min(lossWeight(:,:,:,:,1:(end-1)),lossWeight(:,:,:,:,2:end));  

        eErr_pred_aStart=meshObj_DSpostCell{nt_start}.get([],[],{'stErrPred'},tStart);
        erdErr_pred_aStart=eErr_pred_aStart+meshObj_DSpostCell{nt_start}.get([],[],{'csErrPred'},tStart);
        errAll_init=Poll_aStart(:,:,:,:,1)-analysis_aStart(:,:,:,:,1);

        % do the forecast
        [DSinAstart_test_e,~]=model_AdvErd(u_aStart,v_aStart,Poll_aStart,eErr_pred_aStart,errAll_init,gridWeight_predict);
        [DSinAstart_test_erd,~]=model_AdvErd(u_aStart,v_aStart,Poll_aStart,erdErr_pred_aStart,errAll_init,gridWeight_predict);
        [DSinAstart_test_ietm,~]=model_AdvErd(u_aStart,v_aStart,Poll_aStart,0.*Poll_aStart,errAll_init,gridWeight_predict);

        % calculate the loss
        lossByLead_st{nt_start}=calLossSeries(DSinAstart_test_e,analysis_aStart,lossWeight);
        lossByLead_erd{nt_start}=calLossSeries(DSinAstart_test_erd,analysis_aStart,lossWeight);
        lossByLead_ori{nt_start}=calLossSeries(Poll_aStart,analysis_aStart,lossWeight);
        lossByLead_ietm{nt_start}=calLossSeries(DSinAstart_test_ietm,analysis_aStart,lossWeight);
        lossByLead_st{nt_start}(~qt2evalute)=nan;lossByLead_erd{nt_start}(~qt2evalute)=nan;lossByLead_ori{nt_start}(~qt2evalute)=nan;lossByLead_ietm{nt_start}(~qt2evalute)=nan;
        % write to file
        meshObj_DSpostCell{nt_start}.put(DSinAstart_test_e,[],[],{'ConPred_st'},tStart);
        meshObj_DSpostCell{nt_start}.put(DSinAstart_test_erd,[],[],{'ConPred_erd'},tStart);
        meshObj_DSpostCell{nt_start}.put(DSinAstart_test_ietm,[],[],{'ConPred_ietm'},tStart);
    WaitMessage.Send;
    end
    WaitMessage.Destroy;
end

%% functions for figures
function plotImproveAlongDistribute(yPredict,yTarget)
    yyaxis left
    BinEdges=-200:5:200;
    hpred=histogram(yPredict,BinEdges);
    counts=hpred.Values;
    x2plot=(BinEdges(1:end-1)+BinEdges(2:end))/2;
    % plot(x2plot,counts,'-*')
    ylabel('Count')
    set(gca,'YScale','log');
    yyaxis right
    ry2plot=nan(size(counts));
    fun_rmse=@(ori,pred) sqrt(mean((ori-pred).^2));
    for n_bin=1:numel(ry2plot)
        q=yPredict>=BinEdges(n_bin) & yPredict<=BinEdges(n_bin+1);
        rmse_ori=fun_rmse(yTarget(q),0);
        rmse_curr=fun_rmse(yTarget(q),yPredict(q));
        ry2plot(n_bin)=(rmse_curr-rmse_ori)/rmse_ori;
    end
    bar(x2plot,ry2plot*100)
    ylim([-100 0])
    ylabel('Changes in RMSE (%)')
    xlabel('Predicted value by model_CS')
end