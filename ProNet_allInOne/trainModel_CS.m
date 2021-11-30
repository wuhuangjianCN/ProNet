function [parameters,hf1]=trainModel_myRDmodel(dlX,dlY_Target,parameters,option)
    if nargin>3 && isfield(option,'plotTraining');  plotTraining = option.plotTraining;else; plotTraining = false;end
    if nargin>3 && isfield(option,'testData');  plotTesting = true;else; plotTesting = false;end
    if nargin>3 && isfield(option,'maxEpoch');  maxEpoch = option.maxEpoch;else; maxEpoch = 500;end
    if nargin>3 && isfield(option,'num_partion');  num_partion = option.num_partion;else; num_partion = 1;end
    if nargin>3 && isfield(option,'validationFrequency');  validationFrequency = option.validationFrequency;else; validationFrequency = 1;end
    if nargin>3 && isfield(option,'h_trainingModel');  h_trainingModel = option.h_trainingModel;else; h_trainingModel = @model_deepRD;end
    if nargin>3 && isfield(option,'h_testingModel');  h_testingModel = option.h_testingModel;else; h_testingModel = h_trainingModel;end
    
    trailingAvg = [];trailingAvgSq = [];
    if plotTraining
        hf1=figure;
        ax1=axes(hf1);
        lineLossTrain = animatedline('Color',[0 0.447 0.741],'Marker','o','LineStyle','none');
        xlabel("Iteration");    ylabel('Loss for model_rd');
        grid on;ylim([0 inf]);
        if plotTesting
            lineLossTest = animatedline('Color','r','Marker','.','LineStyle','none');
            legend(ax1,'Train','Test')
        end
    end
    % ######################################## Loop over epochs ##########################################
    WaitMessage = parfor_wait(num_partion*ceil(maxEpoch),'Waitbar',true,'Title','Training model_r');
    x_point = 1;tic_start=tic;
    [XCell_train,YCell_train]=dataPartion(num_partion,dlX,dlY_Target);
    [XCell_test,YCell_test]=dataPartion(num_partion,option.testData.dlX,option.testData.dlY_Target);
    for epoch=1:maxEpoch        
        if plotTesting && (mod(epoch,validationFrequency) == 0 )
            lossTestAll=nan(size(YCell_test));
            for nCell=1:numel(YCell_test)
                dlX_curr=gpuArray(XCell_test{nCell});
                dlY_curr=gpuArray(YCell_test{nCell});
                Y_testPredict=h_testingModel(dlX_curr,parameters);
                lossTestAll(nCell)=mean((Y_testPredict-dlY_curr).^2);
            end
            addpoints(lineLossTest,x_point,double(sqrt(mean(lossTestAll))))
            drawnow
        end
        [~,indStart_rand]=sort(rand(size(YCell_train)));
        for n_ind =1:numel(indStart_rand)
            dlX_curr=XCell_train{indStart_rand(n_ind)};dlY_curr=YCell_train{indStart_rand(n_ind)};
            dlX_curr=gpuArray(dlX_curr);dlY_curr=gpuArray(dlY_curr);
            [gradients,loss,dlY_predict] = dlfeval(@myGradients,dlX_curr,dlY_curr, parameters,h_trainingModel);
            [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
                trailingAvg,trailingAvgSq,x_point);
%             [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
%                 trailingAvg,trailingAvgSq,x_point,0.001,0.5,0.5);
            x_point = x_point + 1;
            if plotTraining 
                addpoints(lineLossTrain,x_point,double(extractdata(sqrt(loss))))
                D = duration(0,0,toc(tic_start),'Format','hh:mm:ss');
                title(ax1,"Epoch: " + epoch + ", Elapsed: " + string(D))
                drawnow
            end
    WaitMessage.Send;
        end
    end
    WaitMessage.Destroy;
end
function [XCell,YCell]=dataPartion(num_partion,Xin,Yin)
    XCell=cell(1,num_partion);YCell=cell(1,num_partion);
    xbars=linspace(0,1,num_partion+1);
    xRand=rand(1,numel(Yin));
    for n_partion=1:num_partion
        q_curr=xRand >= xbars(n_partion) & xRand < xbars(n_partion+1);
        XCell{n_partion}=Xin(:,q_curr);
        YCell{n_partion}=Yin(:,q_curr);
    end
end

function [gradients,loss,dlY] = myGradients(dlX,dlY_target,parameters,h_trainingModel)
    dlY = h_trainingModel(dlX,parameters);
    loss = mean((dlY-dlY_target).^2);
    gradients = dlgradient(loss,parameters);
end


% function dlY = myDeepNN_training(dlX,parameters)
%     % dlX is a 3-d image sequence: CTB, 
%     % where h, w, and c correspond to the height, width, and number of channels of the images respectively, N is the number of observations, and S is the sequence length.
%     dlY=0.*dlX(1,:,:);
%     for nBatch=1:size(dlX,3)
%         tmp=dlX(:,:,nBatch);
%         for n_layer=1:numel(parameters)
%             % fully connect
%             tmp=fullyconnect(tmp,parameters{n_layer}.Weights,parameters{n_layer}.Bias,'DataFormat','CTB');
%             if n_layer < numel(parameters);tmp=relu(tmp);end
%             % tanh
%         end
%         dlY(:,:,nBatch)=tmp(:);
%     end
% end
% 
% function dlY = myDeepNN_testing(dlX,parameters)
%     % dlX is a 3-d image sequence: CTB, 
%     % where h, w, and c correspond to the height, width, and number of channels of the images respectively, N is the number of observations, and S is the sequence length.
%     dlY=0.*dlX(1,:,:);
%     for nBatch=1:size(dlX,3)
%         tmp=dlX(:,:,nBatch);
%         for n_layer=1:numel(parameters)
%             % fully connect
%             tmp=fullyconnect(tmp,parameters{n_layer}.Weights,parameters{n_layer}.Bias,'DataFormat','CTB');
%             if n_layer < numel(parameters);tmp=relu(tmp);end
%             % tanh
%         end
%         dlY(:,:,nBatch)=tmp(:);
%     end
% end
