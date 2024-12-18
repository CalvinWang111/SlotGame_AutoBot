import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import random

def merge_mask(masks, tolerance = 10):
    # The tolerance is in pixels
    merged_masks=[]
    merged_indices = set()

    for i in range(len(masks)):
        if i in merged_indices: continue
        bbox    = masks[i]['bbox']
        current_left   = int(bbox[0])
        current_top    = int(bbox[1])
        current_right  = int(bbox[0] + bbox[2])
        current_buttom = int(bbox[1] + bbox[3])

        for j in range(i+1,len(masks)):
            if j in merged_indices: continue
            bbox        = masks[j]['bbox']
            next_left   = int(bbox[0])
            next_top    = int(bbox[1])
            next_right  = int(bbox[0] + bbox[2])
            next_buttom = int(bbox[1] + bbox[3])

            if (( abs(current_top - next_top)       < tolerance and
                  abs(current_buttom - next_buttom) < tolerance and
                 (abs(current_right - next_left)    < tolerance or abs(current_left - next_right) < tolerance))
                or
                ( abs(current_right - next_right) < tolerance and
                  abs(current_left - next_left)   < tolerance and
                 (abs(current_top - next_buttom)  < tolerance or abs(next_top - current_buttom) < tolerance))):
                
                merged_indices.append(j)
                new_mask=masks[i].copy()
                new_mask['segmentation'][:,:] = masks[i]['segmentation'][:,:]+masks[j]['segmentation'][:,:]
                new_mask['area']              = masks[i]['area'] + masks[j]['area']
                new_mask['bbox'][0]           = min(current_left,next_left)
                new_mask['bbox'][1]           = min(current_top,next_top)
                new_mask['bbox'][2]           = max(current_right,next_right)-new_mask['bbox'][0]
                new_mask['bbox'][3]           = max(current_buttom,next_buttom)-new_mask['bbox'][1]
                # take the arithmetic mean, it might be inaccurate
                new_mask['predicted_iou']     = (masks[i]['predicted_iou'] + masks[j]['predicted_iou'])/2.0
                new_mask['stability_score']   = (masks[i]['stability_score'] + masks[j]['stability_score'])/2.0
                # keep both
                new_mask['point_coords'].append(masks[j]['point_coords'][0])
                
                merged_masks.append(new_mask)
                break
        else: # if can't merge
            merged_masks.append(masks[i])

    return merged_masks

# image_number = 12
# image = cv2.imread('img/source/img{0}.jpg'.format(image_number))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# sam_checkpoint = "segment-anything-main/sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# device = "cuda"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(image)
# print("mask generate completed")
# merged_masks = merge_mask(merge_mask(masks))

# for i in range(len(merged_masks)):
#     mask=merged_masks[i]
#     segmented_image = np.zeros_like(image)
#     segmented_image[mask['segmentation']] = image[mask['segmentation']]
#     cv2.imwrite('img/result/masks/{0}.jpg'.format(i), cv2.cvtColor(segmented_image,cv2.COLOR_BGR2RGB))

# for i in range(len(merged_masks)):
#     bbox=merged_masks[i]['bbox']
#     if(bbox):
#         x1=int(bbox[0])
#         y1=int(bbox[1])
#         x2=int(bbox[0]+bbox[2])
#         y2=int(bbox[1]+bbox[3])
#         color=[(255,0,0),(0,0,255),(0,255,0),(0,0,0),(255,255,255),(0,255,255),(255,255,0),(255,0,255)]
#         ccc = random.randint(0,7)
#         cv2.rectangle(image,(x1,y1),(x2,y2),(color[ccc][0],color[ccc][1],color[ccc][2]),2)
# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# show_anns(merged_masks)
# plt.axis('off')
# plt.savefig('img/result/img{0}.jpg'.format(image_number), bbox_inches='tight')