function [dlY,state] = model_ST(dlX,para,state)
    % dlX is a 2-d image sequence: h-by-w-by-c-by-N-by-S, 
    % where h, w, and c correspond to the height, width, and number of channels of the images respectively, N is the number of observations, and S is the sequence length.
    dlY=0.*dlX(:,:,1,:,:);
    if size(dlX,4)>1;error('size(dlX,4) should be 1');end
    for n_obs=1:size(dlX,4)
        for index_sequence=1:size(dlX,5)
%             disp('model_ST a t')
            dlX_aStep=dlX(:,:,:,n_obs,index_sequence);
            % conv_b1
            dlX_aStep=dlconv(dlX_aStep,para.conv_1.Weights,para.conv_1.Bias,'Padding','same','DataFormat','SSBCT');% same kernal for all channel
            % convLSTM_b1
            [dlX_aStep,state.ConvLSTM_1.H,state.ConvLSTM_1.C]=ConvLSTM(dlX_aStep,state.ConvLSTM_1.H,state.ConvLSTM_1.C,para.ConvLSTM_1);
            % convLSTM_b2
            [~,state.ConvLSTM_2.H,state.ConvLSTM_2.C]=ConvLSTM(dlX_aStep,state.ConvLSTM_1.H,state.ConvLSTM_1.C,para.ConvLSTM_2);
            % conv_b2
            dlX_aStep=dlconv(cat(3,state.ConvLSTM_1.H,state.ConvLSTM_2.H),para.conv_2.Weights,para.conv_2.Bias,'Padding','same','DataFormat','SSCBT');  
            dlY(:,:,:,n_obs,index_sequence)=dlX_aStep;
        end
    end
end