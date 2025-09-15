using Flux,Statistics,Base
export NormPollCell,NormPoll,denorm

mutable struct NormPollCell
    mean::AbstractArray
    std::AbstractArray
    varName::AbstractArray
end
function Base.copy(m::NormPollCell)
    return NormPollCell(copy(m.mean),copy(m.std),copy(m.varName))
end
function NormPoll(data,varName;dim_poll=ndims(data)-1,q_zeroMean=falses(size(data,dim_poll))) #dim_poll默认为最后一个维度之前的一个维度
    dim2cal=(1:ndims(data))[.!((1:ndims(data)).==dim_poll)] #(1,2,4)
    m=mapslices(mean,data,dims=(dim2cal))
    m[q_zeroMean].=0
    stdVar=sqrt.(mapslices(mean,(data .- m).^2,dims=dim2cal))
    return NormPollCell(m ,stdVar,varName )#返回两个向量 (dim_poll,)，(dim_poll,)
end


function (m::NormPollCell)(x::AbstractArray,varName::AbstractArray)
    # 注意！！：norm.mean和。norm.stdVar都要存放在cpu，否者模型运行会报错（ERROR: Scalar indexing is disallowed）

    meanV,stdV=getMeanStdV(m,varName)
    x .= (x .- meanV)./stdV
    return x
end
(m::NormPollCell)(x::AbstractArray,varName::AbstractString)= m(x,[varName])
# set mean and stdVar to be non-trainable by return an empty tupple
# Flux.trainable(m::NormPollCell) = ()
# Flux.@functor NormPollCell # 不能加，否者ERROR: Scalar indexing is disallowed

function denorm(m::NormPollCell, x::AbstractArray,varName::AbstractArray)
    meanV,stdV=getMeanStdV(m,varName)
    x .= x .* stdV .+ meanV
    return x
end
denorm(m::NormPollCell, x::AbstractArray,varName::AbstractString)=denorm(m,x,[varName])

function getMeanStdV(norm::NormPollCell,varName::AbstractArray)    
    if length(norm.mean) > 1
        ind_var=indexin(varName,norm.varName)
        meanV,stdV=norm.mean[ind_var],norm.std[ind_var]
        sizeV=[size(norm.mean)...]
        sizeV[findfirst(sizeV .> 1)]=length(meanV)
        meanV=reshape(meanV,sizeV...)
        stdV=reshape(stdV,sizeV...)
    else
        meanV,stdV=norm.mean,norm.std
    end
    return meanV,stdV
end