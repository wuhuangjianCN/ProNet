classdef normClass <  matlab.mixin.Copyable

	% Copyright: Huangjian Wu   All rights reserved.
    % If you have any problem, please contact me: wuhuangjian@126.com
    
    
    %********************* update ************
    % example usage:
    
    properties
        varsDontChange
        varName_all
        mean_all
        std_all
    end
    
    methods
        function obj=normClass(varsDontChange)
            obj.varsDontChange=varsDontChange;
            obj.varName_all={};
            obj.mean_all=[];
            obj.std_all=[];
        end
        function updateFactor(obj,data,varNamesIn)
            % update the varName and allocate mean and std
            lia_newVar=~ismember(varNamesIn,obj.varName_all);
            varNamesAll=[obj.varName_all,varNamesIn(lia_newVar)];
            obj.varName_all=varNamesAll;
            obj.mean_all=[obj.mean_all,zeros(1,sum(lia_newVar))];
            obj.std_all=[obj.std_all,ones(1,sum(lia_newVar))];
            % update mean and std
            lia_inVar=ismember(varNamesAll,varNamesIn);
            mean_in=mean(data,[1 2 4 5],'omitnan');
            std_in=std(data,0,[1 2 4 5],'omitnan');
            obj.mean_all(lia_inVar)=mean_in(:);
            obj.std_all(lia_inVar)=std_in(:);
            % don't change some var
            lia_noNorm=ismember(obj.varName_all,obj.varsDontChange);
            obj.mean_all(lia_noNorm)=0;
            obj.std_all(lia_noNorm)=1;            
        end
        function data=normData(obj,data,varNamesIn)
            lia_inVar=ismember(obj.varName_all,varNamesIn);
            mean_in=obj.mean_all(lia_inVar);
            std_in= obj.std_all(lia_inVar);
            % In MATLAB® R2016b and later, you can directly use operators instead of bsxfun, since the operators independently support implicit expansion of arrays with compatible sizes.
            lia=ismember(varNamesIn,obj.varName_all);
            data(:,:,lia,:,:)=(data(:,:,lia,:,:)-permute(mean_in(:),[3 2 1]))./permute(std_in(:),[3 2 1]);
        end
        function data=restoreData(obj,data,varNamesIn)
            lia_inVar=ismember(obj.varName_all,varNamesIn);
            mean_in=obj.mean_all(lia_inVar);
            std_in= obj.std_all(lia_inVar);
            lia=ismember(varNamesIn,obj.varName_all);
            % In MATLAB® R2016b and later, you can directly use operators instead of bsxfun, since the operators independently support implicit expansion of arrays with compatible sizes.
            data(:,:,lia,:,:)=data(:,:,lia,:,:).*permute(std_in(:),[3 2 1])+permute(mean_in(:),[3 2 1]);
        end
    end
end