function paraFC=init_netR(numXvar,num_layer,num_width)
    paraFC=cell(num_layer,1);
    paraFC{1}=init_fc(numXvar,num_width);
    for n_layer=2:(num_layer-1);paraFC{n_layer}=init_fc(num_width,num_width);end
    paraFC{end}=init_fc(num_width,1);
end
function fc=init_fc(input_channel,outputFeatures)
    sizeWeight=[outputFeatures,input_channel];
    tmp=randn(sizeWeight)*0.01;
    fc.Weights = dlarray(single(tmp));
    fc.Bias = dlarray(zeros(outputFeatures,1,'single'));
end