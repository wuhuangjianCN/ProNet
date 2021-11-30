function [H1,H2,C] = encoder_ST(dlX,H,C,parameters)
    % dlX is a 2-d image sequence: h-by-w-by-c-by-N-by-S, 
    % where h, w, and c correspond to the height, width, and number of channels of the images respectively, N is the number of observations, and S is the sequence length.
    if size(dlX,4)>1;error('size(dlX,4) should be 1');end
    H2=H;
    for index_sequence=1:size(dlX,5)
        dlX_aStep=dlX(:,:,:,n_obs,index_sequence);
%             % conv_b1
%             dlX_aStep=dlconv(dlX_aStep,parameters.conv_e1.Weights,parameters.conv_e1.Bias,'Padding','same','DataFormat','SSCBT');
        % convLSTM_b1
        [dlX_aStep,H1,C]=ConvLSTM(dlX_aStep,H2,C,parameters.convLSTM_1);
        % convLSTM_b2
        [~,H2,C]=ConvLSTM(dlX_aStep,H1,C,parameters.convLSTM_2);
    end
%     % conv_b2
%     dlX_aStep=dlconv(cat(3,H1,H2),parameters.conv_2.Weights,parameters.conv_2.Bias,'Padding','same','DataFormat','SSCBT');  
%     dlY(:,:,:,n_obs,index_sequence)=dlX_aStep;
end