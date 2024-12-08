from TemplateMatching.grid import BullGrid
from TemplateMatching.symbol_recognizer import *
from TemplateMatching.utils import *

STRAIGHT_VERSION = True
template_match_data = {}
if STRAIGHT_VERSION:
    zoom_rate = 0.5625
else:
    zoom_rate = 1.0

def get_symbol_positions(template_dir,image):    
    process_template_matches(
        template_match_data=template_match_data, 
        template_dir=template_dir, 
        img=image,

        #adaptive 世承建議改法
        zoom_rate = image.shape[0]/1080,

        iou_threshold=0.1*zoom_rate, 
        scale_range=[0.8*zoom_rate, 1.5*zoom_rate],
        scale_step=0.05*zoom_rate,
        threshold=0.95*zoom_rate,
        min_area=5000*zoom_rate,
        border=100*zoom_rate
    )
    
    print('template_match_data',template_match_data)
    matched_positions = []
    for template_name, data in template_match_data.items():
        w, h = data['shape'][1], data['shape'][0]
        for (top_left, scale, _) in data['result']:
            x = top_left[0] + w * scale / 2
            y = top_left[1] + h * scale / 2
            matched_positions.append((x, y))
    if len(matched_positions) == 0:
        print("Could not find any matches")
        return (None,None)
    return matched_positions
    
def recoglize_symbol(img,grid:BullGrid,template_dir,game_mode,cell_border=20):
    for j in range(grid.col):
        row_range = (
            range(grid.row - grid.column_heights[j], grid.row) if game_mode == 'up'
            else range(grid.column_heights[j])
        )
        
        for i in row_range:
            roi = grid.get_roi(i, j)
            x, y, w, h = roi
            x1, x2 = x - cell_border, x + w + cell_border
            y1, y2 = y - cell_border, y + h + cell_border
            cell = img[y1:y2, x1:x2]
            
            symbol_name = process_template_matches(
                template_match_data=template_match_data,
                template_dir=template_dir,
                img=cell,
                iou_threshold=0.1*zoom_rate,
                scale_range=[0.8*zoom_rate, 1.5*zoom_rate],
                scale_step=0.05*zoom_rate,
                threshold=0.8*zoom_rate,
                min_area=5000*zoom_rate,
                match_one=True,
                border=cell_border,
            )
            
            grid[i, j] = symbol_name
    return grid
    