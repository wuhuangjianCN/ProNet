classdef normClassGeneral <  matlab.mixin.Copyable
    % variables should be in the first dimension

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
        function obj=normClassGeneral(varsDontChange)
            obj.varsDontChange=varsDontChange;
            obj.varName_all={};
            obj.mean_all=[];
            obj.std_all=[];
        end
        function updateFactor(obj,data,varNamesIn)
            % variables should be in the first dimension
            % update the varName and allocate mean and std
            lia_newVar=~ismember(varNamesIn,obj.varName_all);
            varNamesAll=[obj.varName_all,varNamesIn(lia_newVar)];
            obj.varName_all=varNamesAll;
            obj.mean_all=[obj.mean_all,zeros(1,sum(lia_newVar))];
            obj.std_all=[obj.std_all,ones(1,sum(lia_newVar))];
            % update mean and std
            lia_inVar=ismember(varNamesAll,varNamesIn);
            data_2d=reshape(data,size(data,1),[]);
            mean_in=mean(data_2d,2,'omitnan');
            std_in=std(data_2d,0,2,'omitnan');
            obj.mean_all(lia_inVar)=mean_in(:);
            obj.std_all(lia_inVar)=std_in(:);
            % don't change some var
            lia_noNorm=ismember(obj.varName_all,obj.varsDontChange);
            obj.mean_all(lia_noNorm)=0;
            obj.std_all(lia_noNorm)=1;            
        end
        function data=normData(obj,data,varNamesIn)
            sizeIn=size(data);
            data_2d=reshape(data,size(data,1),[]);
            lia_inVar=ismember(obj.varName_all,varNamesIn);
            mean_in=obj.mean_all(lia_inVar);
            std_in= obj.std_all(lia_inVar);
            % In MATLAB® R2016b and later, you can directly use operators instead of bsxfun, since the operators independently support implicit expansion of arrays with compatible sizes.
            lia=ismember(varNamesIn,obj.varName_all);
            data_2d(lia,:)=(data_2d(lia,:)-mean_in(:))./std_in(:);
            data=reshape(data_2d,sizeIn);
        end
        function data=restoreData(obj,data,varNamesIn)
            sizeIn=size(data);
            data_2d=reshape(data,size(data,1),[]);
            lia_inVar=ismember(obj.varName_all,varNamesIn);
            mean_in=obj.mean_all(lia_inVar);
            std_in= obj.std_all(lia_inVar);
            lia=ismember(varNamesIn,obj.varName_all);
            % In MATLAB® R2016b and later, you can directly use operators instead of bsxfun, since the operators independently support implicit expansion of arrays with compatible sizes.
            data_2d(lia,:)=data_2d(lia,:).*std_in(:)+mean_in(:);
            data=reshape(data_2d,sizeIn);
        end
    end
end