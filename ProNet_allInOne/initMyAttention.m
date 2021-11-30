function para=initMyAttention(num_channel,num_grid)
    %num_channel=64;
    para.Whv=dlarray(initializeGlorot(num_channel, num_channel));
    para.Whk=dlarray(initializeGlorot(num_channel, num_channel));
    para.Whq=dlarray(initializeGlorot(num_channel, num_channel));
    para.Wmv=dlarray(initializeGlorot(num_channel, num_channel));
    para.Wmk=dlarray(initializeGlorot(num_channel, num_channel));
    para.Wmq=dlarray(initializeGlorot(num_channel, num_channel));
    
    para.Wz=dlarray(initializeGlorot(num_channel, 2*num_channel));
    
    para.Wzi=dlarray(initializeGlorot(num_channel, num_channel));
    para.Wzg=dlarray(initializeGlorot(num_channel, num_channel));
    para.Wzo=dlarray(initializeGlorot(num_channel, num_channel));
    para.Whi=dlarray(initializeGlorot(num_channel, num_channel));
    para.Whg=dlarray(initializeGlorot(num_channel, num_channel));
    para.Who=dlarray(initializeGlorot(num_channel, num_channel));
    para.bi=dlarray(zeros(num_channel, 1,'single'));
    para.bg=dlarray(zeros(num_channel, 1,'single'));
    para.bo=dlarray(zeros(num_channel, 1,'single'));
%     para.bi=zeros(num_channel, num_grid,'single');
%     para.bg=zeros(num_channel, num_grid,'single');
%     para.bo=zeros(num_channel, num_grid,'single');
end

function weights = initializeGlorot(numOut, numIn)
    varWeights = sqrt( 6 / (numIn + numOut) );
    weights = varWeights * (2 * rand([numOut, numIn], 'single') - 1);
end