function [para,state,config]=init_netST(sizeGrid,num_gridRef)

    % ********************** configuration ***********************
    % --------------- for emission  --------------
    % configuration for conv_1
    config.conv_1.input_channel = 3;
    config.conv_1.num_filter = config.conv_1.input_channel;
    config.conv_1.kernel_size = [3 3];
    
    % configuration for SAM_1
    config.SAM_1.channel = 64;
    % configuration for convLSTM_b1
    config.convLSTM_1.input_channel = config.conv_1.num_filter;
    config.convLSTM_1.num_filter = 64;
    config.convLSTM_1.h_w = sizeGrid;
    config.convLSTM_1.kernel_size = [3 3];
    
    % configuration for SAM_2
    config.SAM_2.channel = 64;
    % configuration for convLSTM_b2
    config.convLSTM_2.input_channel = config.convLSTM_1.num_filter;
    config.convLSTM_2.num_filter = 64;
    config.convLSTM_2.h_w = sizeGrid;
    config.convLSTM_2.kernel_size = [3 3];
    
    % configuration for conv_b2
    config.conv_2.input_channel = config.convLSTM_1.num_filter+config.convLSTM_2.num_filter;
    config.conv_2.num_filter = 1;
    config.conv_2.kernel_size = [3 3];

    % ********************** state ***********************
    sFun=@(sizeState) zeros(sizeState,'single');
    % state for ConvLSTM_1
        sizeState=[config.convLSTM_1.h_w,config.convLSTM_1.num_filter];
        state.ConvLSTM_1.H=dlarray(sFun(sizeState));
        state.ConvLSTM_1.C=dlarray(sFun(sizeState));
%         state.ConvLSTM_1.M=dlarray(sFun([config.SAM_1.channel, num_gridRef]));
    % state for ConvLSTM_2
        sizeState=[config.convLSTM_2.h_w,config.convLSTM_2.num_filter];
        state.ConvLSTM_2.H=dlarray(sFun(sizeState));
        state.ConvLSTM_2.C=dlarray(sFun(sizeState));
%         state.ConvLSTM_2.M=dlarray(sFun([config.SAM_2.channel, num_gridRef]));

    % ********************** parameters ***********************
    % parameters for conv1
    para.conv_1=init_conv(1,1,config.conv_1.kernel_size);
    %para.conv_1=init_conv(config.conv_1.input_channel,config.conv_1.num_filter,config.conv_1.kernel_size);
    % parameters for convLSTM_b1
    para.ConvLSTM_1=init_convLSTM(config.convLSTM_1.input_channel,config.convLSTM_1.num_filter,config.convLSTM_1.kernel_size,config.convLSTM_1.h_w);
%     para.ConvLSTM_1.SAM=initMyAttention(config.SAM_1.channel);
%     para.ConvLSTM_1.tconv=init_conv(1,1,[10 10]);
    % parameters for convLSTM_b2
    para.ConvLSTM_2=init_convLSTM(config.convLSTM_2.input_channel,config.convLSTM_2.num_filter,config.convLSTM_2.kernel_size,config.convLSTM_2.h_w);
%     para.ConvLSTM_2.SAM=initMyAttention(config.SAM_2.channel);
%     para.ConvLSTM_2.tconv=init_conv(1,1,[10 10]);
    % parameters for conv_b2
    para.conv_2=init_conv(config.conv_2.input_channel,config.conv_2.num_filter,config.conv_2.kernel_size);

end

function convPara=init_conv(input_channel,num_filter,kernel_size)
    convPara.Weights = dlarray(initializeGlorotConv(input_channel,num_filter,kernel_size));
    convPara.Bias = dlarray(zeros(num_filter,1,'single'));
end
function convPara=init_convLSTM(input_channel,num_filter,kernel_size,h_w)
    % parameters for convLSTM
        % intializing weights for C
        sizeC=[h_w,num_filter];
        cFun=@(sizeF) single(randn(sizeF)*0.01);
        convPara.wci=dlarray(cFun(sizeC));  convPara.wcf=dlarray(cFun(sizeC)); convPara.wco=dlarray(cFun(sizeC));
        % intializing b
        sizeB=[1,1,num_filter];
        bFun=@(sizeState) single(zeros(sizeState));
        convPara.bi=dlarray(bFun(sizeB));   convPara.bf=dlarray(bFun(sizeB));            convPara.bc=dlarray(bFun(sizeB));            convPara.bo=dlarray(bFun(sizeB));
        % initializing weights for X and H
        sizeWx=[kernel_size,input_channel,num_filter];
        wxFun=@(sizeWx) single(0.01*randn(sizeWx));
        convPara.wxi=dlarray(wxFun(sizeWx));            convPara.wxf=dlarray(wxFun(sizeWx));            convPara.wxc=dlarray(wxFun(sizeWx));            convPara.wxo=dlarray(wxFun(sizeWx));
        sizeWh=[kernel_size,num_filter,num_filter];
        whFun=@(sizeWh) single(0.01*randn(sizeWh));
        convPara.whi=dlarray(whFun(sizeWh));            convPara.whf=dlarray(whFun(sizeWh));            convPara.whc=dlarray(whFun(sizeWh));            convPara.who=dlarray(whFun(sizeWh));
end

function weights = initializeGlorotConv(input_channel,num_filter,kernel_size)
    sizeWeight=[kernel_size ,input_channel ,num_filter];
    varWeights = sqrt( 6 / (prod(kernel_size)*( input_channel+num_filter)) );
    weights = varWeights * (2 * rand(sizeWeight, 'single') - 1);
end

