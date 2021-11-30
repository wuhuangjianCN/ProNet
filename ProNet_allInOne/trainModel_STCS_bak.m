function [para_st,para_cs,hf1]=trainModel_STCS(xVarName_st,fileName_dlX_stInit,fileName_dlX_st,fileName_dlY_stTarget,fileName_stLossWeight,para_st,state_st,csDataFun,meshCSInCell,para_cs,option)
    if nargin>3 && isfield(option,'plotTraining');  plotTraining = option.plotTraining;else; plotTraining = false;end
    if nargin>3 && isfield(option,'testFileName');  plotTesting = true;else; plotTesting = false;end
    if nargin>3 && isfield(option,'epoch');  numEpochs = option.epoch;else; numEpochs = 10;end
    if nargin>3 && isfield(option,'validationFrequency');  validationFrequency = option.validationFrequency;else; validationFrequency = 1;end

    if plotTraining
        hf1=figure;
        ax1=axes(hf1);
        lineRefTrain = animatedline( 'Color', [0 0.447 0.741],'Marker','*','LineStyle','none');
        lineSTTrain = animatedline('Color',[0 0.447 0.741],'Marker','+','LineStyle','none');
        lineLossTrain = animatedline('Color',[0 0.447 0.741],'Marker','o','LineStyle','none');
        lineRefTest = animatedline( 'Color', [0.85 0.325 0.098],'Marker','*');
        lineSTTest = animatedline( 'Color', [0.85 0.325 0.098],'Marker','+');
        lineLossTest = animatedline( 'Color', [0.85 0.325 0.098],'Marker','o');
        xlabel("Iteration");    ylabel('Loss for model_rd');
        grid on;ylim([0 inf]);
        legend('TrainRef','TrainLossST','TrainLoss','TestRef','TestLossST','TestLoss','Location','southwest');
    end
    x_point_st=1;trailingAvg_st = [];trailingAvgSq_st = [];
    optimizerInfo_cs.trailingAvg = [];optimizerInfo_cs.trailingAvgSq = [];optimizerInfo_cs.x_point=1;
    % prepare for training --------------------------------
    customOrder=[1 2 5 6 4 3];% [lon lat var start lead]
    mesh_dlX_stInit=meshDATA(fileName_dlX_stInit);mesh_dlX_stInit.setCustomDimOrder(customOrder);
    mesh_dlX_st=meshDATA(fileName_dlX_st);mesh_dlX_st.setCustomDimOrder(customOrder);
    mesh_dlY_stTarget=meshDATA(fileName_dlY_stTarget);mesh_dlY_stTarget.setCustomDimOrder(customOrder);
    mesh_stLossWeight=meshDATA(fileName_stLossWeight);mesh_stLossWeight.setCustomDimOrder(customOrder);
    tStarts_training=mesh_dlY_stTarget.tStarts;
    % prepare for testing --------------------------------
    
    % ######################################## Loop over epochs ##########################################
    WaitMessage = parfor_wait(numel(tStarts_training)*ceil(numEpochs) + plotTesting*numel(tStarts_training)*ceil(numEpochs)/validationFrequency,'Waitbar',true,'Title','Training model_st');
    tic_start=tic;
    for epoch = 1:numEpochs
        % testing --------------------------------
        if plotTesting && (mod(epoch,validationFrequency) == 0 )
            testmesh_dlX_stInit=meshDATA(option.testFileName.fileName_dlX_stInit);testmesh_dlX_stInit.setCustomDimOrder(customOrder);
            testmesh_dlX_st=meshDATA(option.testFileName.fileName_dlX_st);testmesh_dlX_st.setCustomDimOrder(customOrder);
            testMesh_dlY_Target=meshDATA(option.testFileName.fileName_dlY_stTarget);testMesh_dlY_Target.setCustomDimOrder(customOrder);
            testMesh_LossWeight=meshDATA(option.testFileName.fileName_stLossWeight);testMesh_LossWeight.setCustomDimOrder(customOrder);
            tStarts_testing=testMesh_dlY_Target.tStarts;
            lossAll=nan(1,numel(tStarts_testing));lossRef=nan(1,numel(tStarts_testing));
            for nt_start=1:numel(tStarts_testing)
                tStart_test=tStarts_testing(nt_start);
                
                dlXtest_init=testmesh_dlX_stInit.get([],[],[],tStart_test);
                dlXtest=testmesh_dlX_st.get([],[],[],tStart_test);
                dlYtest_Target=testMesh_dlY_Target.get([],[],[],tStart_test);
                lossWeightTest=testMesh_LossWeight.get([],[],[],tStart_test);
                dlY_Predicted = modelAstart_ST(xVarName_st,dlXtest_init,dlXtest,para_st,state_st);
                lossRef(nt_start)=myLossForGradient(dlY_Predicted(:,:,:,:,2:end).*0,dlYtest_Target(:,:,:,:,2:end),lossWeightTest(:,:,:,:,2:end)) ;
                lossAll(nt_start)=myLossForGradient(dlY_Predicted(:,:,:,:,2:end),dlYtest_Target(:,:,:,:,2:end),lossWeightTest(:,:,:,:,2:end)) ;
                if isnan(lossAll(nt_start));error('nan occur');end
    WaitMessage.Send;
            end
            addpoints(lineLossTest,x_point_st, double(mean(lossAll,'all')));
            addpoints(lineRefTest,x_point_st, double(mean(lossRef,'all')));
        end
        % training --------------------------------
        [~,indStart_rand]=sort(rand(size(tStarts_training)));
        lossRef_All=nan(size(tStarts_training));lossST_All=nan(size(tStarts_training));lossSTCS_All=nan(size(tStarts_training));
        for n_ind =1:numel(indStart_rand)
            indStart=indStart_rand(n_ind);
            tStart=tStarts_training(indStart);
            
            %csDataFun,meshCSInCell,para_cs,normalizer_CS,q_predict
            if tStart==meshCSInCell{indStart}.tStarts
                dlX_cs=csDataFun(meshCSInCell{indStart});
            else
                error('wrong meshObj for model_CS')
            end
            dlX_st_init=mesh_dlX_stInit.get([],[],[],tStart);
            dlX_st=mesh_dlX_st.get([],[],[],tStart);
            dlY_Target=mesh_dlY_stTarget.get([],[],[],tStart);
            lossWeight=mesh_stLossWeight.get([],[],[],tStart);
            [gradients,lossAccumulated,dlY_Predicted,dlY_stPredicted,optimizerInfo_cs,para_cs] = myGradients_aStart(xVarName_st,dlX_st_init,dlX_st,dlX_cs,dlY_Target,lossWeight,para_st,state_st,para_cs,optimizerInfo_cs);
            % Update the network parameters using the Adam optimizer.
            [para_st,trailingAvg_st,trailingAvgSq_st] = adamupdate(para_st,gradients, trailingAvg_st,trailingAvgSq_st,x_point_st);

            % Display the training progress.
            if plotTraining
                forecatLossFun=@(pred) sqrt(   sum( (pred-dlY_Target(:,:,:,:,2:end)).^2.*lossWeight(:,:,:,:,2:end),'all')/sum(lossWeight(:,:,:,:,2:end),'all')      );
                lossRef=forecatLossFun(0.*dlY_Predicted(:,:,:,:,2:end)) ;lossRef_All(n_ind)=lossRef;
                lossST=forecatLossFun(dlY_stPredicted(:,:,:,:,2:end)) ;lossST_All(n_ind)=lossST;
                lossSTCS=forecatLossFun(dlY_Predicted(:,:,:,:,2:end)) ;lossSTCS_All(n_ind)=lossSTCS;
                D = duration(0,0,toc(tic_start),'Format','hh:mm:ss');
                addpoints(lineSTTrain,x_point_st,double(lossST))
                addpoints(lineLossTrain,x_point_st,double(lossSTCS))
                addpoints(lineRefTrain,x_point_st,double(lossRef))
                %addpoints(lineRefTrain,x_point,double(extractdata(lossRef)))
                title(ax1,{"mean Ref:" + mean(lossRef_All,'omitnan') + " ST:" + mean(lossST_All,'omitnan') + " ST:" + mean(lossSTCS_All,'omitnan'),...
                    "Epoch: " + epoch + ", Elapsed: " + string(D)})
                drawnow
            end
            x_point_st=x_point_st+1;
    WaitMessage.Send;
        end
    end
    WaitMessage.Destroy;
end

function [grad_st,lossAccumulated,dlY_Predicted,dlY_stPredicted,optimizerInfo_cs,para_cs] = myGradients_aStart(xVarName,dlX_st_init,dlX_st,dlX_cs,dlY_Target,lossWeight,para_st,state_st,para_cs,optimizerInfo_cs)
% init ---------------------
%     disp('init...')
    [~,state_st] = model_ST(dlX_st_init,para_st,state_st);
% forecast --------------
    q_before=contains(xVarName,'Before');
    nGradient=0;grad_st=[];lossAccumulated=[];
    dlY_Predicted=nan(size(dlY_Target));
    dlY_stPredicted=nan(size(dlY_Target));
    for nLead=2:size(dlX_st,5)        
        % dlY_Target ------------------
        % prepare part of dlX -------------------------------------------
        dlX_st_curr=dlX_st(:,:,:,:,nLead);
        if nLead>2;dlX_st_curr(:,:,q_before,:,:)=dlY_predict_current;end
        
        lossWeight_curr=lossWeight(:,:,:,:,nLead);
        dlX_cs_curr=dlX_cs(:,:,:,:,nLead);
        % do the neural network part ----------------------------------
        dlX_st_curr=dlarray(dlX_st_curr);dlX_cs_curr=dlarray(dlX_cs_curr);dlY_Target_curr=dlarray(dlY_Target(:,:,:,:,nLead));
        if canUseGPU;    dlX_st_curr=gpuArray(dlX_st_curr);dlX_cs_curr=gpuArray(dlX_cs_curr); dlY_Target_curr=gpuArray(dlY_Target_curr);end
        if 0==sum(lossWeight_curr) % no valid grid for verify
            [dlY_predict_current,state_st,dlY_st_current] = model_STCS(dlX_st_curr,dlX_cs_curr,para_st,state_st,para_cs);
        else
            [tmp_grad_st,grad_cs,state_st,tmp_loss,dlY_predict_current,dlY_st_current] = dlfeval(@myGradients,dlX_st_curr,dlX_cs_curr,dlY_Target_curr, para_st, state_st,para_cs,lossWeight_curr);
            if isnan(tmp_loss);error('nan loss found');end
            if isempty(grad_st)
                grad_st=tmp_grad_st;        lossAccumulated=tmp_loss;
            else
                lossAccumulated=tmp_loss+lossAccumulated;
                grad_st=fieldfun(@(x,y) x+y,grad_st,tmp_grad_st);
            end
            nGradient=nGradient+1;
        end
        
        % Update para_st using the Adam optimizer.
        [para_cs,optimizerInfo_cs.trailingAvg,optimizerInfo_cs.trailingAvgSq] = adamupdate(para_cs,grad_cs, optimizerInfo_cs.trailingAvg,optimizerInfo_cs.trailingAvgSq,optimizerInfo_cs.x_point);
            
        dlY_Predicted(:,:,:,:,nLead)=gather(extractdata(dlY_predict_current));
        dlY_stPredicted(:,:,:,:,nLead)=gather(extractdata(dlY_st_current));
    end
	lossAccumulated=lossAccumulated/nGradient;
    grad_st=fieldfun(@(x) x/nGradient,grad_st);
end

function [grad_st,grad_cs,state_st,loss,dlY,dlY_st] = myGradients(dlX_st,dlX_cs,dlY_target,para_st, state_st,para_cs,lossWeight_curr)
    [dlY,state_st,dlY_st] = model_STCS(dlX_st,dlX_cs,para_st,state_st,para_cs);
    loss = myLossForGradient(dlY,dlY_st,dlY_target,lossWeight_curr);
    [grad_st,grad_cs] = dlgradient(loss,para_st,para_cs);
end
function loss=myLossForGradient(dlY,dlY_st,dlY_target,weight)
    absLossRatio=0.1;
    loss =sum(weight.*(...
        absLossRatio.*abs(dlY_st-dlY_target)+abs(dlY-dlY_target)...
        ).^2,'all')/sum(weight,'all'); % mae 
end

% function [grad_st,grad_cs,state_st,lossAccumulated,dlY_predict,nGradient]=gradient_aStep(dlX_st,dlX_cs,dlY_Target, para_st, state_st,para_cs,lossWeight_curr,nGradient,grad_st,lossAccumulated)
%     if 0==sum(lossWeight_curr) % no valid grid for verify
%         [~,state_st] = model_ST(dlX_st,para_st,state_st);
%         dlY_predict=0.*dlX_st(:,:,1,:,:);
%     else
%         [tmp_grad_st,grad_cs,state_st,tmp_loss,dlY_predict] = dlfeval(@myGradients,dlX_st,dlX_cs,dlY_Target, para_st, state_st,para_cs,lossWeight_curr);
%         if isnan(tmp_loss);error('nan loss found');end
%         if isempty(grad_st)
%             grad_st=tmp_grad_st;        lossAccumulated=tmp_loss;
%         else
%             lossAccumulated=tmp_loss+lossAccumulated;
%             grad_st=fieldfun(@(x,y) x+y,grad_st,tmp_grad_st);
%         end
%         nGradient=nGradient+1;
%     end
% end
