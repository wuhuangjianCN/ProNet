function [para_st,hf1]=trainModel_ST(xVarName_st,fileName_dlX_stInit,fileName_dlX_st,fileName_dlY_stTarget,fileName_stLossWeight,para_st,state_st,option)
    if nargin>3 && isfield(option,'plotTraining');  plotTraining = option.plotTraining;else; plotTraining = false;end
    if nargin>3 && isfield(option,'testFileName');  plotTesting = true;else; plotTesting = false;end
    if nargin>3 && isfield(option,'epoch');  numEpochs = option.epoch;else; numEpochs = 10;end
    if nargin>3 && isfield(option,'validationFrequency');  validationFrequency = option.validationFrequency;else; validationFrequency = 1;end

    trailingAvg = [];trailingAvgSq = [];
    if plotTraining
        hf1=figure;
        ax1=axes(hf1);
        lineLossTrain = animatedline('Color',[0 0.447 0.741],'Marker','o','LineStyle','none');
        lineRefTrain = animatedline( 'Color', [0 0.447 0.741],'Marker','+','LineStyle','none');
        lineLossTest = animatedline( 'Color', [0.85 0.325 0.098],'Marker','o');
        lineRefTest = animatedline( 'Color', [0.85 0.325 0.098],'Marker','+');
        xlabel("Iteration");    ylabel('Loss for model_rd');
        grid on;ylim([0 inf]);
        legend('TrainLoss','TrainRef','TestLoss','TestRef');
    end
    % prepare for training --------------------------------
    customOrder=[1 2 5 6 4 3];% [lon lat var start lead]
    mesh_dlX_eInit=meshDATA(fileName_dlX_stInit);mesh_dlX_eInit.setCustomDimOrder(customOrder);
    mesh_dlX_e=meshDATA(fileName_dlX_st);mesh_dlX_e.setCustomDimOrder(customOrder);
    mesh_dlY_eTarget=meshDATA(fileName_dlY_stTarget);mesh_dlY_eTarget.setCustomDimOrder(customOrder);
    mesh_eLossWeight=meshDATA(fileName_stLossWeight);mesh_eLossWeight.setCustomDimOrder(customOrder);
    tStarts_training=mesh_dlY_eTarget.tStarts;
    % prepare for testing --------------------------------
    
    % ######################################## Loop over epochs ##########################################
    WaitMessage = parfor_wait(numel(tStarts_training)*ceil(numEpochs) + numel(tStarts_training)*ceil(numEpochs)/validationFrequency,'Waitbar',true,'Title','Training model_e');
    x_point=1;tic_start=tic;
    for epoch = 1:numEpochs
        % testing --------------------------------
        if plotTesting && (mod(epoch,validationFrequency) == 0 )
            testMesh_dlX_eInit=meshDATA(option.testFileName.fileName_dlX_stInit);testMesh_dlX_eInit.setCustomDimOrder(customOrder);
            testMesh_dlX_e=meshDATA(option.testFileName.fileName_dlX_st);testMesh_dlX_e.setCustomDimOrder(customOrder);
            testMesh_dlY_eTarget=meshDATA(option.testFileName.fileName_dlY_stTarget);testMesh_dlY_eTarget.setCustomDimOrder(customOrder);
            testMesh_eLossWeight=meshDATA(option.testFileName.fileName_stLossWeight);testMesh_eLossWeight.setCustomDimOrder(customOrder);
            tStarts_testing=testMesh_dlY_eTarget.tStarts;
            lossAll=nan(1,numel(tStarts_testing));lossRef=nan(1,numel(tStarts_testing));
            for nt_start=1:numel(tStarts_testing)
                tStart_test=tStarts_testing(nt_start);
                
                dlXtest_init=testMesh_dlX_eInit.get([],[],[],tStart_test);
                dlXtest=testMesh_dlX_e.get([],[],[],tStart_test);
                dlYtest_Target=testMesh_dlY_eTarget.get([],[],[],tStart_test);
                lossWeightTest=testMesh_eLossWeight.get([],[],[],tStart_test);
                dlY_Predicted = modelAstart_ST(xVarName_st,dlXtest_init,dlXtest,para_st,state_st);
                lossRef(nt_start)=myLoss(dlY_Predicted(:,:,:,:,2:end).*0,dlYtest_Target(:,:,:,:,2:end),lossWeightTest(:,:,:,:,2:end)) ;
                lossAll(nt_start)=myLoss(dlY_Predicted(:,:,:,:,2:end),dlYtest_Target(:,:,:,:,2:end),lossWeightTest(:,:,:,:,2:end)) ;
                if isnan(lossAll(nt_start));error('nan occur');end
    WaitMessage.Send;
            end
            addpoints(lineLossTest,x_point, double(mean(lossAll,'all')));
            addpoints(lineRefTest,x_point, double(mean(lossRef,'all')));
        end
        % training --------------------------------
        [~,indStart_rand]=sort(rand(size(tStarts_training)));
        for n_ind =1:numel(indStart_rand)
            tStart=tStarts_training(indStart_rand(n_ind));
            
            dlX_init=mesh_dlX_eInit.get([],[],[],tStart);
            dlX=mesh_dlX_e.get([],[],[],tStart);
            dlY_Target=mesh_dlY_eTarget.get([],[],[],tStart);
            lossWeight=mesh_eLossWeight.get([],[],[],tStart);
            [gradients,lossAccumulated,dlY_Predicted] = myGradients_aStart(xVarName_st,dlX_init,dlX,dlY_Target,lossWeight,para_st,state_st);
            % Update the network parameters using the Adam optimizer.
            [para_st,trailingAvg,trailingAvgSq] = adamupdate(para_st,gradients, trailingAvg,trailingAvgSq,x_point);

            % Display the training progress.
            if plotTraining
                lossRef=myLoss(dlY_Predicted(:,:,:,:,2:end).*0,dlY_Target(:,:,:,:,2:end),lossWeight(:,:,:,:,2:end)) ;
                D = duration(0,0,toc(tic_start),'Format','hh:mm:ss');
                addpoints(lineLossTrain,x_point,double(extractdata(lossAccumulated)))
                addpoints(lineRefTrain,x_point,double(lossRef))
                %addpoints(lineRefTrain,x_point,double(extractdata(lossRef)))
                title(ax1,"Epoch: " + epoch + ", Elapsed: " + string(D))
                drawnow
            end
            x_point=x_point+1;
    WaitMessage.Send;
        end
    end
end

function [gradients,lossAccumulated,dlY_Predicted] = myGradients_aStart(xVarName,dlX_init,dlX,dlY_Target,lossWeight,para,state)
% init ---------------------
%     disp('init...')
    [~,state] = model_ST(dlX_init,para,state);
% forecast --------------
    q_before=contains(xVarName,'Before');
    nGradient=0;gradients=[];lossAccumulated=[];
    dlY_Predicted=nan(size(dlY_Target));
    for nLead=2:size(dlX,5)        
        % dlY_Target ------------------
        % prepare part of dlX -------------------------------------------
        dlX_e=dlX(:,:,:,:,nLead);
        if nLead>2;dlX_e(:,:,q_before,:,:)=dlY_predict_current;end
        
        dlX_e=dlarray(dlX_e);dlY_Target_aStep=dlarray(dlY_Target(:,:,:,:,nLead));
        lossWeight_curr=lossWeight(:,:,:,:,nLead);
        % do the neural network part ----------------------------------
        if canUseGPU;    dlX_e=gpuArray(dlX_e); dlY_Target_aStep=gpuArray(dlY_Target_aStep);end
%         disp('a step...')
        [gradients,state,lossAccumulated,dlY_predict_current,nGradient]=gradient_aStep(dlX_e,dlY_Target_aStep, para, state,lossWeight_curr,nGradient,gradients,lossAccumulated);
        dlY_Predicted(:,:,:,:,nLead)=gather(extractdata(dlY_predict_current));
    end
	lossAccumulated=lossAccumulated/nGradient;
    gradients=fieldfun(@(x) x/nGradient,gradients);
end

function [gradients,state,lossAccumulated,dlY_predict,nGradient]=gradient_aStep(dlX_e,dlY_Target, parameters, state,lossWeight_curr,nGradient,gradients,lossAccumulated)
    if 0==sum(lossWeight_curr) % no valid grid for verify
        [~,state] = model_ST(dlX_e,parameters,state);
        dlY_predict=0.*dlX_e(:,:,1,:,:);
    else
        [tmp_gradients,state,tmp_loss,dlY_predict] = dlfeval(@myGradients,dlX_e,dlY_Target, parameters, state,lossWeight_curr);
        if isnan(tmp_loss);error('nan loss found');end
        if isempty(gradients)
            gradients=tmp_gradients;        lossAccumulated=tmp_loss;
        else
            lossAccumulated=tmp_loss+lossAccumulated;
            gradients=fieldfun(@(x,y) x+y,gradients,tmp_gradients);
        end
        nGradient=nGradient+1;
    end
end

function [gradients,state,loss,dlY] = myGradients(dlX,dlY_target,parameters, state,lossWeight_curr)
    [dlY,state] = model_ST(dlX,parameters, state);
    loss = myLoss(dlY,dlY_target,lossWeight_curr);
    gradients = dlgradient(loss,parameters);
end
function loss=myLoss(dlY,dlY_target,weight)   
    loss =sum(weight.*(dlY-dlY_target).^2,'all')/sum(weight,'all'); % mae 
end
