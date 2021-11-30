function dlY = model_rd(dlX,parameters,isTraining)
    % dlX is a 3-d image sequence: CTB, 
    % where h, w, and c correspond to the height, width, and number of channels of the images respectively, N is the number of observations, and S is the sequence length.
    dlY=0.*dlX(1,:,:);
    
    dropOutProbability=0.5;
    for nBatch=1:size(dlX,3)
        tmp=dlX(:,:,nBatch);
        for n_layer=1:numel(parameters)
            % fully connect
            tmp=fullyconnect(tmp,parameters{n_layer}.Weights,parameters{n_layer}.Bias,'DataFormat','CTB');
             if n_layer < numel(parameters);tmp=relu(tmp);end
            % dropout layer for training only
            if nargin>2 && isTraining
                if n_layer < numel(parameters)
                    tmp=tmp.*(rand(size(tmp),'gpuArray')>dropOutProbability);tmp=tmp*1/(1-dropOutProbability);
                end
            end
        end
        dlY(:,:,nBatch)=tmp(:);
    end
end