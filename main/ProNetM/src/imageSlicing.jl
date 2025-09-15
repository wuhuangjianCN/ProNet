using Base
export imageSlicingCell,combineSlicing,assignMargin,cutMargin


mutable struct imageSlicingCell
    imageSize::Tuple{Int,Int}
    snipSize::Tuple{Int,Int}
    overlap::Tuple{Int,Int}
    extendedSize::Tuple{Int,Int}
    nSnips::Tuple{Int,Int}
end
function Base.copy(m::imageSlicingCell)
    return imageSlicingCell((m.imageSize...,),(m.snipSize...,),(m.overlap...,),(m.extendedSize...,),(m.nSnips...,))
end
function imageSlicingCell(imageSize,snipSize,overlap)
    # imageSize: size of the image
    # snipSize: size of the snip
    # overlap: overlap between snips
    # return: a list of snips
    extendedSize=Int.(ceil.(imageSize./snipSize).*snipSize .+ 2 .* overlap)
    nSnips=Int.((extendedSize .- 2 .* overlap)./snipSize)
    return imageSlicingCell(imageSize,snipSize,overlap,extendedSize,nSnips)
end
function (m::imageSlicingCell)(imagesIn::AbstractArray{T}) where T
    sizeIn=[size(imagesIn)...]
    imagesIn=reshape(imagesIn,(m.imageSize...,:))
    sizeExtendedSnip=m.snipSize .+ 2 .* m.overlap
    snipOut=Array{T}(undef,(sizeExtendedSnip...,size(imagesIn,3),Int(prod(m.nSnips))))
    iStart_all=1 .+ m.snipSize[1] .*(0:m.nSnips[1]-1)
    jStart_all=1 .+ m.snipSize[2] .*(0:m.nSnips[2]-1)
    extendedTemplate_all=[zeros(T,m.extendedSize) for i in 1:Threads.nthreads()]
    Threads.@threads for n in axes(imagesIn,3)
        # extendedTemplate=view(extendedTemplate_all,:,:,Threads.threadid())
        extendedTemplate=extendedTemplate_all[Threads.threadid()]
        extendedTemplate[m.overlap[1] .+ (1:m.imageSize[1]),m.overlap[2] .+(1:m.imageSize[2])]=imagesIn[:,:,n]
        for (ni,iStart) in enumerate(iStart_all)
            iEnd=iStart + sizeExtendedSnip[1]-1 
            for (nj,jStart) in enumerate(jStart_all)
                jEnd=jStart+sizeExtendedSnip[2]-1
                snipOut[:,:,n,ni+(nj-1)*m.nSnips[1]]=extendedTemplate[iStart:iEnd,jStart:jEnd]
            end
        end
    end
    # shape the outputs, put snips in the last dimension
    sizeOut=sizeIn
    sizeOut[1:2]=[sizeExtendedSnip...]
    if length(sizeOut)==2
        sizeOut=(sizeOut...,Int(prod(m.nSnips)))
    else
        sizeOut[end] = sizeOut[end] * Int(prod(m.nSnips))
    end
    return reshape(snipOut,sizeOut...)
end
function assignMargin(m::imageSlicingCell,snips,value2assign)
    sizeOut=size(snips)
    snips=reshape(snips,(sizeOut[1:2]...,:))
    snips[1:m.overlap[1],:,:] .= value2assign
    snips[end-m.overlap[1]+1:end,:,:] .= value2assign
    snips[:,1:m.overlap[2],:] .= value2assign
    snips[:,end-m.overlap[2]+1:end,:] .= value2assign
    return reshape(snips,sizeOut)
end
function cutMargin(m::imageSlicingCell,snips)
    if ndims(snips)==5
        return snips[m.overlap[1]+1:end-m.overlap[1],m.overlap[2]+1:end-m.overlap[2],:,:,:]
    else
        @error "only support 5D array"
    end
end
function combineSlicing(m::imageSlicingCell,snipIn::AbstractArray{T}) where T
    sizeIn=[size(snipIn)...]
    sizeExtendedSnip=m.snipSize .+ 2 .* m.overlap
    snipIn=reshape(snipIn,(sizeExtendedSnip...,:,Int(prod(m.nSnips))))   
    imagesOut=Array{T}(undef,(m.extendedSize...,size(snipIn,3)))
    iStart_all=m.overlap[1] .+ (1:m.snipSize[1]:m.imageSize[1])
    jStart_all=m.overlap[2] .+ (1:m.snipSize[2]:m.imageSize[2])
    for n in axes(imagesOut,3)
        for (ni,iStart) in enumerate(iStart_all)
            iEnd=iStart + m.snipSize[1]-1 
            for (nj,jStart) in enumerate(jStart_all)
                jEnd=jStart+m.snipSize[2]-1
                imagesOut[iStart:iEnd,jStart:jEnd,n]=snipIn[1+m.overlap[1]:end-m.overlap[1],
                                                          1+m.overlap[2]:end-m.overlap[2],
                                                          n,
                                                          ni+(nj-1)*m.nSnips[1]
                                                          ]
            end
        end
    end
    # shape the outputs
    sizeOut=sizeIn
    sizeOut[1:2]=[m.imageSize...]
    sizeOut[end] = Int.(sizeOut[end] / Int(prod(m.nSnips)))
    imagesOut=reshape(imagesOut[m.overlap[1] .+ (1:m.imageSize[1]) , m.overlap[2] .+ (1:m.imageSize[2]),:],sizeOut...)
    return imagesOut
end

"""
# ---------------- testing ----------------
using GradDataM,Plots
image=read(gradData("/home/kdjf/ProNet_julia/DATA/datagrid/id.ctl"),"id")

imageSize=(465,300)
snipSize=(93,75)
overlap=(1,1)
@time imageSlier=imageSlicingCell(imageSize,snipSize,overlap);
@time snips=imageSlier(image);
@time imageOut=combineSlicing(imageSlier,snips);

# plot
heatmap(imageOut[:,:,1,1]')



ter=read(gradData("/home/kdjf/ProNet_julia/DATA/datagrid/grid.ctl"),"ter")
land=read(gradData("/home/kdjf/ProNet_julia/DATA/datagrid/grid.ctl"),"land")

images=cat(image,ter,land,dims=3)
@time snips=imageSlier(images);
@time imagesOut=combineSlicing(imageSlier,snips);

# plot
heatmap(imagesOut[:,:,3,]')


"""