from PIL import Image,ImageFilter,ImageOps

def getBorder(border,length,image):
    size = image.size

    #image.show()
    if border == "left":
        length = min(length,size[1])
        return image.crop((0,0,length,size[1]))
    elif border == "right":
        length = min(length,size[1])
        return image.crop((size[0]-length,0,size[0],size[1]))
    elif border == "top":
        length = min(length,size[0])
        return image.crop((0,0,size[0],length))
    elif border == "bottom":
        length = min(length,size[0])
        return image.crop((0,size[1]-length,size[0],size[1]))
    raise NameError(border + ' is not a valid border name must be top,bottom,left or right')

def concatImages(images,axis=0):
    widths, heights = zip(*(i.size for i in images))
    offset = 0

    if axis == 1:
        max_width = max(widths)
        total_height = sum(heights)
        new_im = Image.new('RGB', (max_width, total_height))

        for im in images:
            new_im.paste(im, (0,offset))
            offset += im.size[1]
    else:
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        for im in images:
            new_im.paste(im, (offset,0))
            offset += im.size[0]

    return new_im

def mirrorExtend(len,image):
    top = getBorder("top",len,image)
    bottom = getBorder("bottom",len,image)

    tmp = concatImages([ImageOps.flip(top),image,ImageOps.flip(bottom)], axis = 1)

    left = getBorder("left",len,tmp)
    right = getBorder("right",len,tmp)

    return concatImages([ImageOps.mirror(left),tmp,ImageOps.mirror(right)],axis = 0)


def applyFilter(filter,image):
    size = image.size
    kernel_size = filter.filterargs[0][0]
    offset = int(kernel_size/2)
    extended_img = mirrorExtend(offset,image)

    filter_extended_img = extended_img.filter(filter)
    filter_extended_img.show()
    filter_img = filter_extended_img.crop((offset,offset,offset+size[0],offset+size[1]))

    return filter_img
