import cv2
import numpy as np
from pathlib import Path

def custom_nms_boxes(nms_boxes, sorted_confidences, iou_threshold):
    if len(nms_boxes) == 0:
        return []

    indices = []  # List to store the indices of boxes to keep
    boxes = np.array(nms_boxes)
    confidences = np.array(sorted_confidences)

    # Extract coordinates of bounding boxes
    x1 = boxes[:, 0]  # x-coordinate of the top-left corner
    y1 = boxes[:, 1]  # y-coordinate of the top-left corner
    x2 = boxes[:, 2]  # x-coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y-coordinate of the bottom-right corner

    # Compute the area of each bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # create an array of indices from 0 to N-1
    order = np.arange(len(confidences))

    while order.size > 0:
        # Index of the current box with the highest confidence score
        i = order[0]
        indices.append(i)  # Add current index to the list of kept indices
        xx1 = np.maximum(x1[i], x1[order[1:]])  # Max of top-left x-coordinates
        yy1 = np.maximum(y1[i], y1[order[1:]])  # Max of top-left y-coordinates
        xx2 = np.minimum(x2[i], x2[order[1:]])  # Min of bottom-right x-coordinates
        yy2 = np.minimum(y2[i], y2[order[1:]])  # Min of bottom-right y-coordinates

        # Compute the width and height of the intersection rectangles
        w = np.maximum(0, xx2 - xx1 + 1)  # Overlapping width
        h = np.maximum(0, yy2 - yy1 + 1)  # Overlapping height

        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union

        # Identify boxes with IoU less than or equal to the threshold
        inds = np.where(iou <= iou_threshold)[0]

        # Update the order array to process the next set of boxes
        order = order[inds + 1]

    return indices

def apply_nms_and_filter_by_best_scale(matching_results, template_shape, iou_threshold):
    if not matching_results:
        return [], None

    # Find the best scale by keeping only matches with the highest match value
    best_match = max(matching_results, key=lambda x: x[2])  # x[2] is the match_val
    best_scale = best_match[1]

    # Filter out all matches that don't have the best scale
    best_scale_matches = [match for match in matching_results if match[1] == best_scale]

    # Prepare bounding boxes and confidence scores for NMS (using only best scale matches)
    boxes = []
    confidences = []

    for (top_left, scale, match_val) in best_scale_matches:
        h, w = int(template_shape[0] * scale), int(template_shape[1] * scale)
        box = [int(top_left[0]), int(top_left[1]), w, h]  # (x, y, width, height)
        boxes.append(box)
        confidences.append(float(match_val))  # Convert match_val to float for OpenCV

    # Sort bounding boxes by confidence scores (from highest to lowest)
    indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    sorted_boxes = [boxes[i] for i in indices]
    sorted_confidences = [confidences[i] for i in indices]
    sorted_best_scale_matches = [best_scale_matches[i] for i in indices]

    # Convert to required format for NMS: boxes in (x, y, x+w, y+h) format
    nms_boxes = [[x, y, x + w, y + h] for (x, y, w, h) in sorted_boxes]

    # Apply custom NMS
    nms_indices = custom_nms_boxes(nms_boxes, sorted_confidences, iou_threshold)

    filtered_results = []
    if len(nms_indices) > 0:
        for i in nms_indices:
            filtered_results.append(sorted_best_scale_matches[i])

    return filtered_results, best_scale

def template_matching(template, img, scale_range, scale_step, threshold, border, match_one=False):
    scales = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)
    img_area = (img.shape[0] - border) * (img.shape[1] - border)
    padding = 5  # Padding to add around the image for reverse matching

    # Split template into RGB and alpha channel
    b_channel, g_channel, r_channel, alpha_channel = cv2.split(template)
    template_rgb = cv2.merge((b_channel, g_channel, r_channel))
    mask = cv2.threshold(alpha_channel, 16, 255, cv2.THRESH_BINARY)[1]
    
    matching_results = []  # To store the locations of matches
    
    for scale in scales:
        # Resize template and mask for the current scale
        resized_template = cv2.resize(template_rgb, (0, 0), fx=scale, fy=scale)
        resized_mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
        result = None
        
        template_area = resized_template.shape[0] * resized_template.shape[1]
        # if match_one and template_area / img_area < 0.2: # Skip if template area is too small
        #     continue
        
        # Ensure the resized template is not larger than the image
        if resized_template.shape[0] > img.shape[0] or resized_template.shape[1] > img.shape[1]:
            if resized_template.shape[0] >= img.shape[0] - 2*border and resized_template.shape[1] >= img.shape[1] - 2*border:
                # perform reverse matching
                img_h, img_w = img.shape[:2]
                img_without_border = img[border+padding:img_h-border-padding, border+padding:img_w-border-padding]
                result = cv2.matchTemplate(resized_template, img_without_border, cv2.TM_CCORR_NORMED)
            else:
                continue
        else:
            # Perform template matching
            result = cv2.matchTemplate(img, resized_template, cv2.TM_CCORR_NORMED, mask=resized_mask)
        
        # Find locations where the match is greater than the threshold
        loc = np.where(result >= threshold)

        # Collect all the matching points
        for pt in zip(*loc[::-1]):  # Switch x and y in zip
            matching_results.append((pt, scale, result[pt[1], pt[0]])) # (top_left, scale, match_val)
    return matching_results

def process_template_matches(template_match_data, template_dir, img, iou_threshold, scale_range, scale_step, threshold, min_area, border, match_one=False, debug=False):

    max_score = 0
    match_one_template = None
    match_one_template_shape = None
    match_one_filtered_results = None
    match_one_scale = None
    
    # Iterate through each template in the folder
    for path in template_dir.glob('*.png'):  # Assuming templates are PNG files
        template_name = path.stem
        template = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # Load template as RGBA
        template_shape = template.shape  # Get template shape for NMS

        # Check if this template already has a best scale in the template_match_data
        if template_name in template_match_data and template_match_data[template_name]['best_scale'] is not None:
            best_scale = template_match_data[template_name]['best_scale']
            matching_results = template_matching(template, img, scale_range=[best_scale, best_scale], scale_step=1.0, threshold=threshold, border=border, match_one=match_one)
        else:
            matching_results = template_matching(template, img, scale_range=scale_range, scale_step=scale_step, threshold=threshold, border=border, match_one=match_one)
        
        # Apply NMS and filter by best scale
        filtered_results, best_scale = apply_nms_and_filter_by_best_scale(matching_results, template_shape, iou_threshold=iou_threshold)
        
        # Check if the area of the bounding box is less than the minimum area
        if (best_scale is not None) and (template_shape[0] * best_scale) * (template_shape[1] * best_scale) < min_area:
            best_scale = None
            filtered_results = []
            
        # Skip if no matches found
        if best_scale is None or not filtered_results:
            continue
        
        # Add or update the result in the dictionary
        if not match_one:
            if template_name not in template_match_data:
                template_match_data[template_name] = {
                    'shape': template_shape,
                    'result': filtered_results,
                    'best_scale': best_scale
                }
            else:
                template_match_data[template_name]['result'] = filtered_results
        
        elif match_one:
            if max_score < filtered_results[0][2]:
                if debug:
                    print(f"Match found for {template_name} with score {filtered_results[0][2]}")
                max_score = filtered_results[0][2]
                match_one_template = template_name
                match_one_template_shape = template_shape
                match_one_filtered_results = filtered_results
                match_one_scale = best_scale

        
    if match_one == True:
        template_match_data[match_one_template] = {
            'shape': match_one_template_shape,
            'result': match_one_filtered_results,
            'best_scale': match_one_scale
        }
        return match_one_template
    
def get_grid_info(points, tolerance=30):
    # tolerance is in pixel
    def get_cluster_center(values):       
        grouped_indices = set()
        cluster_center = []
        
        while True:
            reference_values = []
            for i in range(len(values)):
                if i in grouped_indices:
                    continue
                reference_values.append(values[i])
                grouped_indices.add(i)
                break
            else: break
            
            grouping_completed = False
            while not grouping_completed:
                grouping_completed = True
                i = 0
                while i < len(reference_values):
                    for j in range(len(values)):
                        if j not in grouped_indices and abs(reference_values[i] - values[j]) < tolerance:
                            reference_values.append(values[j])
                            grouped_indices.add(j)
                            grouping_completed = False
                    i += 1

            cluster_center.append(sum(reference_values) / len(reference_values))

        cluster_center.sort()
        return cluster_center
    
    col_x = get_cluster_center([point[0] for point in points])
    row_y = get_cluster_center([point[1] for point in points])

    col_width = []
    for i in range(len(col_x)-1):
        col_width.append(col_x[i+1] - col_x[i])
    grouped_col_width = min(get_cluster_center(col_width))

    avg_col_width = grouped_col_width
    
    row_height = []
    for i in range(len(row_y)-1):
        row_height.append(row_y[i+1] - row_y[i])
    avg_row_height = min(get_cluster_center(row_height))

    display_x      = int(min(col_x) - avg_col_width/2)
    display_y      = int(min(row_y) - avg_row_height/2)
    display_width  = int(max(col_x) + avg_col_width/2 - display_x)
    display_height = int(max(row_y) + avg_row_height/2 - display_y)
    m = round((max(row_y)-min(row_y))/avg_row_height+1)
    n = round((max(col_x)-min(col_x))/avg_col_width+1)
    
    return (display_x, display_y, display_width, display_height), (m,n)

def draw_bboxes_and_icons_on_image(img, template_dir, grid, save_path, icon_size=50):
    color = (255, 255, 255)
    
    for i in range(grid.row):
        for j in range(grid.col):
            if grid[i, j] is None:
                continue
            template_name = grid[i, j]
            template_path = template_dir / f"{template_name}.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
            template_rgb = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)  # Convert to RGB, discard alpha
            resized_template = cv2.resize(template_rgb, (icon_size, icon_size))
            
            # Draw the bounding box
            x, y, w, h = grid.get_roi(i, j)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 5)
            
            icon_top_left = (x, y)
            icon_bottom_right = (x + resized_template.shape[1], y + resized_template.shape[0])
            img[icon_top_left[1]:icon_bottom_right[1], icon_top_left[0]:icon_bottom_right[0]] = resized_template
    
    cv2.imwrite(str(save_path), img)