import torch, random, numpy as np

# Generate the coordinates of a random bounding box based on the mixing ratio 'lam'
def rand_bbox(size, lam):
    W,H = size[2], size[3]
    cut_rat = np.sqrt(1.-lam)
    cut_w, cut_h = int(W*cut_rat), int(H*cut_rat)
    cx, cy = random.randint(0,W), random.randint(0,H)
    bbx1 = np.clip(cx - cut_w//2, 0, W)
    bby1 = np.clip(cy - cut_h//2, 0, H)
    bbx2 = np.clip(cx + cut_w//2, 0, W)
    bby2 = np.clip(cy + cut_h//2, 0, H)
    return bbx1,bby1,bbx2,bby2

# Mix two images by cutting a patch from one and pasting it over another
def cutmix(x,y,alpha=1.0):
    # Determine mixing proportion from a Beta distribution
    lam = np.random.beta(alpha,alpha)
    rand_index = torch.randperm(x.size(0)).to(x.device)

    # Generate random bounding box coordinates for the patch
    y_a, y_b = y, y[rand_index]
    bbx1,bby1,bbx2,bby2 = rand_bbox(x.size(), lam)

    # Replace the patch in the original images with the patch from shuffled images
    x[:,:,bby1:bby2,bbx1:bbx2] = x[rand_index,:,bby1:bby2,bbx1:bbx2]

    # Adjust the lambda value based on the final patch area
    lam = 1-((bbx2-bbx1)*(bby2-bby1)/(x.size(-1)*x.size(-2)))

    # Return the mixed image and original labels for loss calculation
    return x, y_a, y_b, lam 


   