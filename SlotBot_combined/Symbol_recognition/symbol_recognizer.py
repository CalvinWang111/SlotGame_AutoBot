import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from .grid import BaseGrid
from .template import Template

def custom_nms_boxes(nms_boxes, sorted_confidences, iou_threshold):
    """
    進行自訂的 NMS (Non-Maximum Suppression)，
    根據 IoU 閾值過濾重疊過大的框。
    """
    if len(nms_boxes) == 0:
        return []

    # 用於記錄最終保留之框索引的List
    indices = []
    boxes = np.array(nms_boxes)
    confidences = np.array(sorted_confidences)

    # 取出各框的左上角(x1, y1)與右下角(x2, y2)座標
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 計算每個 bounding box 的面積
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 建立與 confidences 長度相同的索引陣列，從 0 ~ N-1
    order = np.arange(len(confidences))

    while order.size > 0:
        # 取得當前信心度最高的框索引
        i = order[0]
        indices.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 計算重疊區塊的寬與高
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union

        # 只保留 IoU 小於等於閾值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return indices

def apply_nms_and_filter_by_best_scale(matching_results, template_shape, iou_threshold):
    """
    從所有縮放比的匹配結果中，挑選最佳縮放比，
    並應用 NMS 過濾重複框，最終回傳篩選後的結果及該縮放比。
    """
    if not matching_results:
        return [], None

    # 先找出匹配值最高的那筆，以決定最佳縮放比
    best_match = max(matching_results, key=lambda x: x[2])  # x[2] 為 match_val
    best_scale = best_match[1]

    # 只保留與最佳縮放比相符的結果
    best_scale_matches = [match for match in matching_results if match[1] == best_scale]

    # 準備進行 NMS 所需要的資料 (bounding boxes 與信心度)
    boxes = []
    confidences = []

    for (top_left, scale, match_val) in best_scale_matches:
        h, w = int(template_shape[0] * scale), int(template_shape[1] * scale)
        box = [int(top_left[0]), int(top_left[1]), w, h]  # (x, y, width, height)
        boxes.append(box)
        confidences.append(float(match_val))

    # 先依照信心度排序，從高到低
    indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    sorted_boxes = [boxes[i] for i in indices]
    sorted_confidences = [confidences[i] for i in indices]
    sorted_best_scale_matches = [best_scale_matches[i] for i in indices]

    # 轉換為 NMS 需要的格式 (x, y, x+w, y+h)
    nms_boxes = [[x, y, x + w, y + h] for (x, y, w, h) in sorted_boxes]

    # 進行自訂 NMS
    nms_indices = custom_nms_boxes(nms_boxes, sorted_confidences, iou_threshold)

    filtered_results = []
    if len(nms_indices) > 0:
        for i in nms_indices:
            filtered_results.append(sorted_best_scale_matches[i])

    return filtered_results, best_scale

def template_matching(template, roi: np.ndarray, iou_threshold, scale_range, scale_step, threshold, min_area, border, grayscale=False):
    """
    使用 OpenCV matchTemplate() 進行模板匹配，
    並依序嘗試指定的縮放範圍來尋找最佳結果。
    """
    template_shape = template.shape

    # 若 scale_range 的最小值與最大值相同，表示固定縮放比
    if scale_range[0] == scale_range[1]:
        scales = [scale_range[0]]
    else:
        scales = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)

    # padding：在反向匹配時使用，預設 5
    padding = 5

    # 先將 template 拆分 RGB 與 alpha，再合併成普通 BGR 或灰階
    b_channel, g_channel, r_channel, alpha_channel = cv2.split(template)
    template = cv2.merge((b_channel, g_channel, r_channel))
    mask = cv2.threshold(alpha_channel, 250, 255, cv2.THRESH_BINARY)[1]
    if grayscale:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    matching_results = []

    for scale in scales:
        # 依當前 scale 重設 template 與 mask 大小
        resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        resized_mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
        result = None
        
        # 確認當前縮放後之模板不會超出整個 ROI 的範圍
        if resized_template.shape[0] > roi.shape[0] or resized_template.shape[1] > roi.shape[1]:
            # 若模板大小近似整個 ROI，嘗試執行反向匹配
            if resized_template.shape[0] >= roi.shape[0] - 2*border and resized_template.shape[1] >= roi.shape[1] - 2*border:
                try:
                    img_h, img_w = roi.shape[:2]
                    img_without_border = roi[border+padding : img_h-border-padding,
                                             border+padding : img_w-border-padding]
                    result = cv2.matchTemplate(resized_template, img_without_border, cv2.TM_CCORR_NORMED)
                except:
                    continue
            else:
                continue
        else:
            # 一般情況下的模板匹配
            result = cv2.matchTemplate(roi, resized_template, cv2.TM_CCORR_NORMED, mask=resized_mask)
        
        # 找出符合 threshold 以上的所有區域
        loc = np.where(result >= threshold)

        # 收集這些匹配區域的 (top_left 座標, scale, match_val)
        for pt in zip(*loc[::-1]):
            matching_results.append((pt, scale, result[pt[1], pt[0]]))
            
    # 使用前面定義的 NMS 流程並挑選最佳縮放比
    filtered_results, match_scale = apply_nms_and_filter_by_best_scale(matching_results, template_shape, iou_threshold=iou_threshold)
    
    # 若最終沒有任何匹配，或縮放比無效，直接回傳 None
    if match_scale is None or not filtered_results:
        return None, None, None

    # 若計算出的 bounding box 面積小於 min_area，也視為無效
    if (template_shape[0] * match_scale) * (template_shape[1] * match_scale) < min_area:
        return None, None, None
        
    match_score = filtered_results[0][2]
    return match_scale, match_score, filtered_results

def process_template_matches(template_list: List[Template], roi: np.ndarray,
                             iou_threshold, scale_range, scale_step, threshold,
                             min_area, border, grayscale=False, match_one=False, debug=False):
    """
    對多個 Template 依序進行 template_matching()，
    依需求決定是否只取匹配度最高的單一結果 (match_one)，
    或將所有匹配到的位置通通回傳。
    """
    max_score = 0
    match_one_template_obj = None
    match_one_scale = None
    matched_positions = []
    
    for template_obj in template_list:
        template_name = template_obj.name
        template = template_obj.img
        tempalte_shape = template.shape

        # 若此模板已經有最佳縮放比，直接用固定 scale 重新匹配
        if template_obj.best_scale is not None:
            scale = template_obj.best_scale
            match_scale, match_score, filtered_results = template_matching(
                template=template,
                roi=roi,
                iou_threshold=iou_threshold,
                scale_range=[scale, scale],
                scale_step=1.0,
                threshold=threshold,
                min_area=min_area,
                border=border,
                grayscale=grayscale
            )
        else:
            match_scale, match_score, filtered_results = template_matching(
                template=template,
                roi=roi,
                iou_threshold=iou_threshold,
                scale_range=scale_range,
                scale_step=scale_step,
                threshold=threshold,
                min_area=min_area,
                border=border,
                grayscale=grayscale
            )
        
        if match_scale is None or match_score is None or filtered_results is None:
            continue
        
        # 若 match_one 為 False，則記錄所有符號匹配位置；並更新模板的最佳縮放比與分數
        if not match_one:
            if template_obj.best_scale is None:
                template_obj.best_scale = match_scale
                template_obj.match_score = match_score
            # 取得所有匹配點的中心位置 (x, y)，可用於盤面分析或畫網格
            for (top_left, scale, _) in filtered_results:
                matched_positions.append(
                    (top_left[0] + tempalte_shape[1] * scale / 2 + border,
                     top_left[1] + tempalte_shape[0] * scale / 2 + border)
                )
                if debug:
                    # 在 ROI 影像上畫出方框以做除錯顯示
                    roi = cv2.rectangle(
                        roi, 
                        top_left, 
                        (int(top_left[0] + tempalte_shape[1] * scale), int(top_left[1] + tempalte_shape[0] * scale)), 
                        (0, 255, 0), 2
                    )
        # 若 match_one 為 True，則只關心最高分的模板
        elif match_one:
            if max_score < match_score:
                max_score = match_score
                match_one_template_obj = template_obj
                match_one_scale = match_scale

        if debug:
            print(f'{template_name:<20} | score: {match_score:.3f} | scale: {match_scale:.3f}')

    # 若只取單一匹配，可能在除錯時視需要顯示最終結果
    if debug and match_one and match_one_template_obj:
        matched_copy = match_one_template_obj.img
        matched_copy = cv2.resize(matched_copy, (0, 0), fx=match_one_scale, fy=match_one_scale)
        cv2.imshow(match_one_template_obj.name, matched_copy)
        cv2.imshow('ROI', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("-----------------------------------")

    if match_one:
        if match_one_template_obj is not None:
            match_one_template_obj.best_scale = match_one_scale
            return match_one_template_obj, max_score
        return None, None
    else:
        return matched_positions

def process_template_matches_sift(template_list: List[Template], roi: np.ndarray,
                                  scale_range, min_matches, ratio_threshold,
                                  ransac_threshold, vertical_threshold,
                                  horizontal_threshold, debug=False):
    """
    以 SIFT 特徵為基礎的匹配方式，適用於模板匹配難以處理的場景。
    會先建立 SIFT 特徵與描述子，再比對目標影像與模板是否相符。
    """
    # 建立 SIFT 偵測器與 BFMatcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    best_match = None
    best_num_matches = 0
    best_keypoints = None
    best_template_keypoints = None
    best_template_img = None
    best_matches = None
    best_scale = None
    best_score = float('inf')
    best_obj = None

    # 使用 CLAHE 增強 ROI 的影像對比度，方便特徵偵測
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    target_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    target_gray = clahe.apply(target_gray)
    target_height, target_width = target_gray.shape[:2]
    
    # 尋找 ROI 的特徵點與描述子
    keypoints_target, descriptors_target = sift.detectAndCompute(target_gray, None)

    # 若找不到足夠特徵點，則無法進行 SIFT 比對
    if descriptors_target is None or len(keypoints_target) < 2:
        if debug:
            print("目標 ROI 沒有足夠特徵點，無法進行 SIFT。")
        return None, None

    # 依序處理每個模板
    for template_obj in template_list:
        template = template_obj.img
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 同樣也對模板做 CLAHE 增強
        template_gray = clahe.apply(template_gray)
        template_height, template_width = template_gray.shape[:2]
        
        # 偵測模板的特徵點與描述子
        keypoints_template, descriptors_template = sift.detectAndCompute(template_gray, None)
        if descriptors_template is None or len(keypoints_template) < 2:
            continue

        # 進行 KNN 比對，並使用 Lowe's Ratio Test 過濾
        matches_knn = bf.knnMatch(descriptors_template, descriptors_target, k=2)
        good_matches = []
        for m, n in matches_knn:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        # 若透過此初步篩選後，仍有足夠特徵點則可繼續
        if len(good_matches) >= min_matches:
            src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches])

            # 先將特徵點位置正規化 (除以寬高)，檢查垂直與水平誤差
            src_pts_norm = src_pts / [template_width, template_height]
            dst_pts_norm = dst_pts / [target_width, target_height]

            position_diffs = src_pts_norm - dst_pts_norm
            vertical_diffs = position_diffs[:, 1]
            horizontal_diffs = position_diffs[:, 0]

            # 過濾掉在垂直或水平方向偏移太大的特徵
            position_consistent_matches = []
            for i, m in enumerate(good_matches):
                if abs(vertical_diffs[i]) <= vertical_threshold and abs(horizontal_diffs[i]) <= horizontal_threshold:
                    position_consistent_matches.append(m)

            # 若還有足夠的一致性特徵，嘗試進行 RANSAC 估計仿射變換
            if len(position_consistent_matches) >= 4:
                src_pts_consistent = np.float32([keypoints_template[m.queryIdx].pt for m in position_consistent_matches]).reshape(-1, 1, 2)
                dst_pts_consistent = np.float32([keypoints_target[m.trainIdx].pt for m in position_consistent_matches]).reshape(-1, 1, 2)

                M, mask_ransac = cv2.estimateAffinePartial2D(src_pts_consistent, dst_pts_consistent,
                                                             method=cv2.RANSAC,
                                                             ransacReprojThreshold=ransac_threshold)
                if M is not None:
                    matches_mask = mask_ransac.ravel().tolist()
                    filtered_matches = [m for m, keep in zip(position_consistent_matches, matches_mask) if keep]

                    if len(filtered_matches) >= min_matches:
                        # 統計總距離 (distance) 以作為一種分數，越小越好
                        score = sum([m.distance for m in filtered_matches])

                        # 從仿射矩陣估計 X 與 Y 方向的縮放比，再取平均
                        scale_x = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
                        scale_y = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
                        estimated_scale = (scale_x + scale_y) / 2

                        # 檢查是否在允許的縮放範圍內
                        min_scale, max_scale = scale_range
                        if min_scale <= estimated_scale <= max_scale:
                            num_filtered_matches = len(filtered_matches)
                            # 更新目前最好的匹配結果（考慮匹配數量及 score 高低）
                            if (num_filtered_matches > best_num_matches) or \
                               (num_filtered_matches == best_num_matches and score < best_score):
                                best_match = template_obj.name
                                best_num_matches = num_filtered_matches
                                best_keypoints = keypoints_target
                                best_template_keypoints = keypoints_template
                                best_template_img = template_gray
                                best_matches = filtered_matches
                                best_scale = estimated_scale
                                best_score = score
                                best_obj = template_obj
                            if debug:
                                print(f'{template_obj.name:<20} | Matches: {num_filtered_matches:<5} | Scale: {estimated_scale:<6.2f} | Score: {score:<6.2f}')
                        else:
                            if debug:
                                print(f'{template_obj.name:<20} | 估計縮放比 {estimated_scale:5.2f} 超出範圍')
                    else:
                        if debug:
                            print(f'{template_obj.name:<20} | RANSAC 過濾後特徵點不足 ({len(filtered_matches)})')
                else:
                    if debug:
                        print(f'{template_obj.name:<20} | RANSAC 仿射轉換失敗')
            else:
                if debug:
                    print(f'{template_obj.name:<20} | 符合一致性的特徵點不足 ({len(position_consistent_matches)})')
        else:
            if debug:
                print(f'{template_obj.name:<20} | 初步可用特徵點不足 ({len(good_matches)})')

    if not best_match:
        if debug:
            print("未找到任何可用的 SIFT 匹配。")
        return None, None

    # 視覺化：可選，除錯或顯示用
    keypoints_target_img = cv2.drawKeypoints(
        target_gray,
        best_keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    keypoints_template_img = cv2.drawKeypoints(
        best_template_img,
        best_template_keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    matches_img = cv2.drawMatches(
        best_template_img,
        best_template_keypoints,
        target_gray,
        best_keypoints,
        best_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    if debug:
        cv2.imshow("Target Keypoints", keypoints_target_img)
        cv2.imshow("Template Keypoints", keypoints_template_img)
        cv2.imshow("Filtered Matches", matches_img)
        print(f'最佳匹配: {best_obj.name} | 配對特徵數: {best_num_matches} | 縮放比: {best_scale:.2f} | 分數: {best_score:<10.2f}')
        print("-----------------------------------")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return best_obj, best_score
    
def get_grid_info(points, tolerance=30):
    """
    根據盤面上散落的座標點（多為符號中心點），
    以群集方式找出整體網格的 x、y 座標，並推論網格大小及列行數。
    tolerance：像素允許誤差，用於判斷座標是否屬於同一群。
    """
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
            else:
                break
            
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
    for i in range(len(col_x) - 1):
        col_width.append(col_x[i+1] - col_x[i])
    grouped_col_width = min(get_cluster_center(col_width))

    avg_col_width = grouped_col_width
    
    row_height = []
    for i in range(len(row_y) - 1):
        row_height.append(row_y[i+1] - row_y[i])
    avg_row_height = min(get_cluster_center(row_height))

    display_x      = int(min(col_x) - avg_col_width/2)
    display_y      = int(min(row_y) - avg_row_height/2)
    display_width  = int(max(col_x) + avg_col_width/2 - display_x)
    display_height = int(max(row_y) + avg_row_height/2 - display_y)
    m = round((max(row_y) - min(row_y)) / avg_row_height + 1)
    n = round((max(col_x) - min(col_x)) / avg_col_width + 1)
    
    return (display_x, display_y, display_width, display_height), (m, n)

def draw_grid_on_image(img, grid: BaseGrid, color=(255, 255, 255), thickness=5):
    """
    根據 grid 內部 row、col 的定義，
    在輸入影像上將每個 cell 的區域畫出外框，方便除錯或視覺化。
    """
    for i in range(grid.row):
        for j in range(grid.col):
            x, y, w, h = grid.get_roi(i, j)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    return img

def draw_bboxes_and_icons_on_image(img, template_dir, grid, save_path, icon_size=35):
    """
    在影像中畫出網格的框線與辨識到的圖示 (icon)，
    最後輸出存檔至 save_path。
    """
    color = (255, 255, 255)
    for i in range(grid.row):
        for j in range(grid.col):
            if grid[i, j] is None or grid[i, j]["symbol"] is None:
                continue
            template_name = grid[i, j]["symbol"]
            template_path = template_dir / f"{template_name}.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
            template_rgb = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
            resized_template = cv2.resize(template_rgb, (icon_size, icon_size))
            
            # 先畫 bounding box
            x, y, w, h = grid.get_roi(i, j)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 5)
            
            # 接著將符號小圖貼於 cell 左上角
            icon_top_left = (x, y)
            icon_bottom_right = (x + resized_template.shape[1], y + resized_template.shape[0])
            img[icon_top_left[1]:icon_bottom_right[1], icon_top_left[0]:icon_bottom_right[0]] = resized_template
    
    cv2.imwrite(str(save_path), img)
