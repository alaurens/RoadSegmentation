import numpy as np
import math  
import imageProcess as ip



def train_generator(imgs,label,batch_size):

    batch_imgs = np.zeros((batch_size,400,400,3))
    batch_label = np.zeros((batch_size,400,400,1))
    while True: 
        for i in range (batch_size):
            index = np.random.choice(len(imgs),1)
            batch_imgs[i] = imgs[index]
            batch_label[i] = label[index]
            yield batch_imgs,batch_label
            
def get_patch(img,patch_dim):
    num_channels = img.shape[2]
    size = np.size(img, 0)
    dim = (0,patch_dim,patch_dim,num_channels)
    patches = [] 
    for i in range (0,size,patch_dim):
        for j in range (0,size,patch_dim):
            patch = img[i:i+patch_dim,j:j+patch_dim,:]  
            
            patches.append(patch)
    
    patches = np.asarray(patches)
    return patches
            
def patch_generator(test_image, patch_dim):

    if np.size(test_image, 0) % patch_dim == 0:
         test_image = test_image
    else:
        test_image = ip.numpy2pillow(test_image)
        add_pixel = patch_dim*(np.floor(np.size(test_image, 0) / patch_dim )+1) - np.size(test_image, 0)
        test_image = ip.mirror_extend(add_pixel/2, test_image)
        test_image = ip.pillow2numpy(test_image)
           
    vec = get_patch(test_image,patch_dim)
    
    return vec 

def prediction_generator(prediction_patch):
    
    number_patch = (prediction_patch.shape[0])
    len_patch = math.sqrt(number_patch)
    
    for i in range (0,prediction_patch.shape[0],len_patch):
        for j in range (0,prediction_patch.shape[0],len_patch):
            img[i,j] = prediction_patch[j+i*len_patch]
    
    img = ip.numpy2pillow(img)
    
    return img