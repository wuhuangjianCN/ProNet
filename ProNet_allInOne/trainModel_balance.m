function [para_b,hf1]=trainModel_balance(xVarName,fileName_dlX_bInit,fileName_dlX_b,fileName_ErrAllTarget,fileName_other,q_grid2predict,para_b,state_b,option)
    if nargin>3 && isfield(option,'plotTraining');  plotTraining = option.plotTraining;else; plotTraining = false;end
    if nargin>3 && isfield(option,'testFileName');  plotTesting = true;else; plotTesting = false;end
    if nargin>3 && isfield(option,'epoch');  numEpochs = option.epoch;else; numEpochs = 10;end
    if nargin>3 && isfield(option,'validationFrequency');  validationFrequency = option.validationFrequency;else; validationFrequency = 1;end

    trailingAvg = [];trailingAvgSq = [];
    if plotTraining
        hf1=figure('Position',[97   487  1035   420]);
        %ax1=axes(hf1);
        ax1=axes('Position',[0.074396        0.11     0.51111       0.815]);
        lineLossTrain = animatedline('Color',[0 0.447 0.741],'Marker','o','LineStyle','none');
        lineRefTrain = animatedline( 'Color', [0 0.447 0.741],'Marker','+','LineStyle','none');
        lineLossTest = animatedline( 'Color', [0.85 0.325 0.098],'Marker','o');
        lineRefTest = animatedline( 'Color', [0.85 0.325 0.098],'Marker','+');
        xlabel("Iteration");    ylabel('Loss for model_balance','Interpreter','none');
        grid on;ylim([0 inf]);
        legend('TrainLoss','TrainRef','TestLoss','TestRef');
        ax2=axes('Position',[0.63991    0.095714     0.33466       0.815]);
        
        if plotTesting
            hf_testByLead=figure('Position',[1187   487   560   420]);
            ax_testByLead=axes(hf_testByLead);hold(ax_testByLead,'on');grid(ax_testByLead,'on')
        end
    end
    % prepare for training --------------------------------
    customOrder=[1 2 5 6 4 3];% [lon lat var start lead]
    mesh_dlX_eInit=meshDATA(fileName_dlX_bInit);mesh_dlX_eInit.setCustomDimOrder(customOrder);
    mesh_dlX_e=meshDATA(fileName_dlX_b);mesh_dlX_e.setCustomDimOrder(customOrder);
    mesh_ErrAll_Target=meshDATA(fileName_ErrAllTarget);mesh_ErrAll_Target.setCustomDimOrder(customOrder);
    mesh_other=meshDATA(fileName_other);mesh_other.setCustomDimOrder(customOrder);
    tStarts_training=mesh_ErrAll_Target.tStarts;
    % prepare for testing --------------------------------
    
    % ######################################## Loop over epochs ##########################################
    WaitMessage = parfor_wait(numel(tStarts_training)*ceil(numEpochs) + plotTesting* numel(tStarts_training)*ceil(numEpochs)/validationFrequency,'Waitbar',true,'Title','Training model_balance');
    x_point=1;tic_bart=tic;
    for epoch = 1:numEpochs
        % testing --------------------------------
        if plotTesting && (mod(epoch,validationFrequency) == 1 )
            testMesh_dlX_eInit=meshDATA(option.testFileName.fileName_dlX_bInit);testMesh_dlX_eInit.setCustomDimOrder(customOrder);
            testMesh_dlX_e=meshDATA(option.testFileName.fileName_dlX_b);testMesh_dlX_e.setCustomDimOrder(customOrder);
            testMesh_ErrAll_Target=meshDATA(option.testFileName.fileName_bErrAllTarget);testMesh_ErrAll_Target.setCustomDimOrder(customOrder);
            testMesh_bOther=meshDATA(option.testFileName.fileName_bOther);testMesh_bOther.setCustomDimOrder(customOrder);
            tStarts_testing=testMesh_ErrAll_Target.tStarts;
            lossAll=nan(1,numel(tStarts_testing));lossRef=nan(1,numel(tStarts_testing));
            madCell_byLead_ds=cell(1,numel(tStarts_testing));
            parfor nt_start=1:numel(tStarts_testing)
                tStart=tStarts_testing(nt_start);
                
                % prepare data
                dlX_init=testMesh_dlX_eInit.get([],[],[],tStart);
                dlX_fore=testMesh_dlX_e.get([],[],[],tStart);
                ErrAll_Target=testMesh_ErrAll_Target.get([],[],[],tStart);
                lossWeight=testMesh_bOther.get([],[],'lossWeight',tStart);
                u_aStart=testMesh_bOther.get([],[],'u',tStart);
                v_aStart=testMesh_bOther.get([],[],'v',tStart);
                con_ori_aStart=testMesh_bOther.get([],[],'con_ori',tStart);
                [bErr_Predicted,ErrAll_Predicted] = modelAstart_Balance(xVarName,dlX_init,dlX_fore,ErrAll_Target(:,:,:,:,1),u_aStart,v_aStart,con_ori_aStart,q_grid2predict,para_b,state_b);
                lossRef(nt_start)=myLoss(ErrAll_Target(:,:,:,:,2:end).*0,ErrAll_Target(:,:,:,:,2:end),lossWeight(:,:,:,:,2:end)) ;
                lossAll(nt_start)=myLoss(ErrAll_Predicted(:,:,:,:,2:end),ErrAll_Target(:,:,:,:,2:end),lossWeight(:,:,:,:,2:end)) ;
                
                lossByLeadFun=@(target,pred) squeeze(sqrt(mean(((target-pred).*lossWeight).^2,[ 1 2 3 4])));
                madCell_byLead_ds{nt_start}=lossByLeadFun(ErrAll_Target,ErrAll_Predicted);
                if isnan(lossAll(nt_start));error('nan occur');end
    WaitMessage.Send;
            end
            addpoints(lineLossTest,x_point, double(mean(lossAll,'all')));
            addpoints(lineRefTest,x_point, double(mean(lossRef,'all')));
            mad_byLead_ds=cell2mat(madCell_byLead_ds);
            plot(ax_testByLead,mean(mad_byLead_ds,2,'omitnan'));legend(ax_testByLead);drawnow;
        end
        % training --------------------------------
        [~,indStart_rand]=sort(rand(size(tStarts_training)));lossRef_All=nan(size(tStarts_training));lossBalance_All=nan(size(tStarts_training));
        for n_ind =1:numel(indStart_rand)
            tStart=tStarts_training(indStart_rand(n_ind));
            % prepare data
            dlX_init=mesh_dlX_eInit.get([],[],[],tStart);
            dlX_fore=mesh_dlX_e.get([],[],[],tStart);
            ErrAll_Target=mesh_ErrAll_Target.get([],[],[],tStart);
            lossWeight=mesh_other.get([],[],'lossWeight',tStart);
            u_aStart=mesh_other.get([],[],'u',tStart);
            v_aStart=mesh_other.get([],[],'v',tStart);
            con_ori_aStart=mesh_other.get([],[],'con_ori',tStart);
            
            % do the gradient
            [gradients,lossAccumulated,ErrAll_Predicted] = myGradients_aStart(xVarName,dlX_init,dlX_fore,ErrAll_Target,u_aStart,v_aStart,con_ori_aStart,lossWeight,q_grid2predict,para_b,state_b);
            % Update the network parameters using the Adam optimizer.
            [para_b,trailingAvg,trailingAvgSq] = adamupdate(para_b,gradients, trailingAvg,trailingAvgSq,x_point);

            % Display the training progress.
            if plotTraining
                forecatLossFun=@(pred) sqrt(   sum( (pred-ErrAll_Target(:,:,:,:,2:end)).^2.*lossWeight(:,:,:,:,2:end),'all')/sum(lossWeight(:,:,:,:,2:end),'all')      );
                % plot loss --------------------
                lossRef=forecatLossFun(ErrAll_Target(:,:,:,:,2:end).*0) ;lossRef_All(n_ind)=lossRef;
                lossBalance=forecatLossFun(ErrAll_Predicted(:,:,:,:,2:end)) ;lossBalance_All(n_ind)=lossBalance;
                D = duration(0,0,toc(tic_bart),'Format','hh:mm:ss');
                addpoints(lineLossTrain,x_point,double(lossBalance))
                addpoints(lineRefTrain,x_point,double(lossRef))
                %addpoints(lineRefTrain,x_point,double(extractdata(lossRef)))
                title(ax1,"mean Ref:" + mean(lossRef_All,'omitnan') + " Balance:" + mean(lossBalance_All,'omitnan')+...
                    "   Epoch: " + epoch + ", Elapsed: " + string(D))
%                 title(ax1,{"mean Ref:" + mean(lossRef_All,'omitnan') + " Balance:" + mean(lossBalance_All,'omitnan'),...
%                     "Epoch: " + epoch + ", Elapsed: " + string(D)})
                drawnow
                % plot loss by lead ---------------------
                lossByLeadFun=@(target,pred) squeeze(sqrt(mean(((target-pred).*lossWeight).^2,[ 1 2 3 4])));
                cla(ax2)
                data2plot=[lossByLeadFun(ErrAll_Target,0.*ErrAll_Target),lossByLeadFun(ErrAll_Target,ErrAll_Predicted)];
                plot(ax2,0:(size(data2plot,1)-1),data2plot);legend('Original','Corrected');xlabel('lead (hour)');ylabel('RMSE')
                drawnow
            end
            x_point=x_point+1;
    WaitMessage.Send;
        end
    end
    WaitMessage.Destroy;
end

function [gradients,lossAccumulated,ErrAll_Predicted] = myGradients_aStart(xVarName,dlX_init,dlX_fore,ErrAll_Target,u_aStart,v_aStart,con_ori_aStart,lossWeight,q_grid2predict,para,state)
    dx=15e3;dy=15e3;
    lead2startGradient=-24;
    leadsAll_init=(-size(dlX_init,5)+1):0;
% init ------------------------------------------------------
    q_leads=leadsAll_init<lead2startGradient;
    dlX_curr=dlarray(dlX_init(:,:,:,:,q_leads));
    if canUseGPU;    dlX_curr=gpuArray(dlX_curr); end
    [~,state] = model_ST(dlX_curr,para,state);
        
    meanLossWeight=mean(lossWeight,5);
% gradient before forecast ---------------------------------
    nGradient=0;gradients=[];lossAccumulated=[];
    nLead2spin_all=find(leadsAll_init>=lead2startGradient);
    for n_spin=1:numel(nLead2spin_all)
        nLead2spin=nLead2spin_all(n_spin);
        dlX_curr=dlX_init(:,:,:,:,nLead2spin);
        dlY_Target_curr=0.*dlX_curr(:,:,1,:,:);
        dlX_curr=dlarray(dlX_curr); dlY_Target_curr=dlarray(dlY_Target_curr);
        if canUseGPU;    dlX_curr=gpuArray(dlX_curr);dlY_Target_curr=gpuArray(dlY_Target_curr); end
        [gradients,state,lossAccumulated,~,nGradient]=gradient_aStep(dlX_curr,dlY_Target_curr, para, state,meanLossWeight,nGradient,gradients,lossAccumulated);
    end

% gradient durinig forecast ------------------------------
    q_advErr=contains(xVarName,'advErr');
    q_erdErr=contains(xVarName,'erdErrPredicted');
    erdErrPredicted=dlX_fore(:,:,q_erdErr,:,:);
    errPre=ErrAll_Target(:,:,:,:,1);
    ErrAll_Predicted=nan(size(ErrAll_Target),'single');   ErrAll_Predicted(:,:,:,:,1) =ErrAll_Target(:,:,:,:,1);
    for nLead=2:size(dlX_fore,5)
        % transport the get the erdErr_Target,advErr --------------------
        con_ori=con_ori_aStart(:,:,:,:,nLead);
        u=u_aStart(:,:,:,:,nLead);
        v=v_aStart(:,:,:,:,nLead);
        advErr=advDiffAnHour_wuhj(errPre, u, v, dx, dy);
        % prepare dlY_Target -------------------------------------------
        dlY_Target_curr=ErrAll_Target(:,:,:,:,nLead)-advErr-erdErrPredicted(:,:,:,:,nLead);    
        dlY_Target_curr=dlarray(dlY_Target_curr); 
        
        % prepare dlX -------------------------------------------------
        dlX_curr=dlX_fore(:,:,:,:,nLead);
        dlX_curr(:,:,q_advErr,:,:)=advErr;
        lossWeight_curr=lossWeight(:,:,:,:,nLead);
        % do the neural network part ----------------------------------
        dlX_curr=dlarray(dlX_curr); dlY_Target_curr=dlarray(dlY_Target_curr);
        if canUseGPU;    dlX_curr=gpuArray(dlX_curr); dlY_Target_curr=gpuArray(dlY_Target_curr);end
        [gradients,state,lossAccumulated,dlY_predict_curr,nGradient]=gradient_aStep(dlX_curr,dlY_Target_curr, para, state,lossWeight_curr,nGradient,gradients,lossAccumulated);
        dlY_predict_curr=gather(extractdata(dlY_predict_curr));
        
        % postprocess for the next iteration --------------------
        % add limit to ensure the nonagative of predicted concentration:
        erdb_aSetp=erdErrPredicted(:,:,:,:,nLead)+dlY_predict_curr;
        erdb_aSetp(~q_grid2predict)=0;
        errAll_aStep=advErr+erdb_aSetp;
        errAll_aStep([1 end],:)=0;errAll_aStep(:,[1 end])=0;
        q_exceed=con_ori-errAll_aStep<0;
        errAll_aStep(q_exceed)=-con_ori(q_exceed);                
        errPre=errAll_aStep;
        
        ErrAll_Predicted(:,:,:,:,nLead)=errPre;
    end
	lossAccumulated=lossAccumulated/nGradient;
    gradients=fieldfun(@(x) x/nGradient,gradients);
end

function [gradients,state,lossAccumulated,dlY_predict,nGradient]=gradient_aStep(dlX_e,dlY_Target, para, state,lossWeight_curr,nGradient,gradients,lossAccumulated)
    if 0==sum(lossWeight_curr) % no valid grid for verify
        [~,state] = model_ST(dlX_e,para,state);
        dlY_predict=0.*dlX_e(:,:,1,:,:);
    else
        [tmp_gradients,state,tmp_loss,dlY_predict] = dlfeval(@myGradients,dlX_e,dlY_Target, para, state,lossWeight_curr);
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

function [gradients,state,loss,dlY] = myGradients(dlX,dlY_target,para, state,lossWeight_curr)
    [dlY,state] = model_ST(dlX,para, state);
    loss = myLoss(dlY,dlY_target,lossWeight_curr);
    gradients = dlgradient(loss,para);
end
function loss=myLoss(dlY,dlY_target,weight)   
    loss =sum(weight.*(dlY-dlY_target).^2,'all')/sum(weight,'all'); % mae 
end
