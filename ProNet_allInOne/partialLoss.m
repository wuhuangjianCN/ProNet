function loss=partialLoss(dlY,dlY_target,weight)
    %loss = sum(weight.*mean(abs(mean(dlY(:,:,:,:,qt)-dlY_target(:,:,:,:,qt),5)),[3 4]),'all')/sum(weight,'all');
    %loss = sum(weight.*mean(abs(dlY(:,:,:,:,qt)-dlY_target(:,:,:,:,qt)),[3 4 5]),'all')/sum(weight,'all'); % mae
    %loss =sum(weight.*mean(abs(dlY(:,:,:,:,qt))+abs(dlY(:,:,:,:,qt)-dlY_target(:,:,:,:,qt)),[3 4 5]),'all')/sum(weight,'all'); % mae with panalty
    %loss =sum(weight.*mean(abs(dlY-dlY_target),[3 4 5]),'all')/sum(weight,'all'); % mae with panalty
    loss =sum(weight.*abs(dlY-dlY_target),'all')/sum(weight,'all'); % mae 
    %loss = sqrt(sum(weight.*mean((dlY(:,:,:,:,qt)-dlY_target(:,:,:,:,qt)).^2,[3 4 5]),'all')/sum(weight,'all')); % mse
    %loss =meanParameters(parameters)*100*2+mean(abs(dlY(:,:,:,:,qt)),'all')+ sum(weight.*mean(abs(dlY(:,:,:,:,qt)-dlY_target(:,:,:,:,qt)),[3 4 5]),'all')/sum(weight,'all'); % mae with panalty
    %loss = sum(weight.*mean(abs(dlY(:,:,:,:,nt2evaluate)-dlY_target(:,:,:,:,nt2evaluate)),[3 4 5]),'all')/sum(weight,'all');
    %loss = sqrt(mean((dlY(:,:,:,:,nt2evaluate)-dlY_target(:,:,:,:,nt2evaluate)).^2,'all'));
end