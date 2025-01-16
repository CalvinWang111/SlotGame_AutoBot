import cv2
import numpy as np
from pathlib import Path
import json
import time
from .grid import *
from .template import *
from .symbol_recognizer import *
from .utils import *
from typing import List, Dict

# 設定根目錄為當前檔案往上三層目錄
root_dir = Path(__file__).parent.parent.parent

class BaseGridRecognizer:
    def __init__(self, game: str, mode: str, config_file: Path, window_size=(1920, 1080), debug=False):
        """
        BaseGridRecognizer 為遊戲盤面辨識的基礎類別。
        會依照指定的 config 檔案與遊戲模式，載入模板、建立或載入網格，
        並在需要時進行盤面初始化與符號辨識。
        """
        self.game = game
        self.mode = mode
        self.debug = debug
        # 讀取外部設定檔 (JSON)，取得各項路徑及參數
        self.config = self.load_config(config_file)

        # 組合出實際運作時需要使用的路徑
        self.template_dir = root_dir / Path(self.config["template_dir"].format(game=game, mode=mode))
        self.save_dir = root_dir / Path(self.config["save_dir"].format(game=game, mode=mode))
        self.grid_path = root_dir / Path(self.config["grid_path"].format(game=game, mode=mode))
        self.output_json_dir = root_dir / Path(self.config["output_json_dir"].format(game=game, mode=mode))

        # 確保輸出目錄存在
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.output_json_dir.mkdir(parents=True, exist_ok=True)
        
        # 決定當前遊戲畫面的橫直向，並依此計算縮放比
        self.layout_orientation = self.config["layout_orientation"]
        self.window_size = window_size
        horizontal_base_size = (1920, 1080)
        vertical_base_size = (1080, 1920)

        # 若為 horizontal，將寬度做為判斷基準；若為 vertical，則以高度做為判斷基準
        if self.layout_orientation == "horizontal":
            self.adjustment_ratio = window_size[0] / horizontal_base_size[0]
        elif self.layout_orientation == "vertical":
            self.adjustment_ratio = window_size[1] / vertical_base_size[1]
        print(f'Adjustment ratio: {self.adjustment_ratio}')

        # 依照縮放比調整部分參數，如 min_area、border 等
        self.cell_border = int(self.config["cell_border"] * self.adjustment_ratio)

        self.grid_matching_params = self.config["grid_matching_params"]
        self.grid_matching_params['min_area'] = int(self.grid_matching_params['min_area'] * (self.adjustment_ratio ** 2))
        self.grid_matching_params['border'] = int(self.grid_matching_params['border'] * self.adjustment_ratio)

        self.cell_matching_params = self.config["cell_matching_params"]
        self.cell_matching_params['min_area'] = int(self.cell_matching_params['min_area'] * (self.adjustment_ratio ** 2))
        self.cell_matching_params['border'] = int(self.cell_matching_params['border'] * self.adjustment_ratio)

        self.sift_matching_params = self.config["sift_matching_params"]
        
        # 決定初始化或辨識格子時，是使用 color 或 gray 模式
        self.initialize_grid_mode = self.config["initialize_grid_mode"]
        self.cell_matching_mode = self.config["cell_matching_mode"]

        # 決定辨識模式：'template only','template first','sift first' 三種
        method_list = ['template only','template first','sift first']
        cell_matching_method = self.config.get('cell_matching_method', 'sift first')
        self.cell_matching_method = method_list.index(cell_matching_method)
        
        # cell_size：盤面預期每個格子的寬高
        self.cell_size = self.config["cell_size"]
        self.cell_size[0] = int(self.cell_size[0] * self.adjustment_ratio)
        self.cell_size[1] = int(self.cell_size[1] * self.adjustment_ratio)
        print(f'Cell size: {self.cell_size}')

        # non_square_scale：針對非方形模板額外再乘上的縮放因子
        self.non_square_scale = self.config["non_square_scale"]
        
        self.square_templates: List[Template] = []
        self.non_square_templates: List[Template] = []
        self.all_templates: List[Template] = []
        
        self.resize_template = self.config.get('resize_template', True)

        # 載入符號模板
        self.load_templates()

        # 若有設定要依照 cell_size 進行模板縮放，則執行
        if self.resize_template:
            self.resize_templates_by_cell_size()
        
        # 是否要嘗試載入先前已儲存的盤面 (pkl)
        self.use_saved_grid = self.config.get('use_saved_grid', True)
        self.grid = None
        if self.use_saved_grid:
            self.load_grid()
        if self.grid is None:
            self.grid_is_stabilized = False
        else:
            self.grid_is_stabilized = True
            
        # 用於暫存一些模板辨識的額外資料
        self.template_match_data = {}
        self.debug = debug
        
    def load_templates(self):
        """
        讀取指定 template_dir 下的所有 PNG 檔，建立 Template 物件。
        若透明度較低（透明像素比例 < 0.2），視為方形樣板；否則視為非方形。
        最終將所有樣板放到 all_templates。
        """
        print(str(self.template_dir.absolute()))
        for path in self.template_dir.glob('*.png'):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            alpha_channel = img[:, :, 3]
            total_pixels = alpha_channel.size
            transparent_pixels = np.sum(alpha_channel < 32)
            transparent_ratio = transparent_pixels / total_pixels
            template_obj = Template(path, img)

            if transparent_ratio < 0.2:
                template_obj.is_square = True
                self.square_templates.append(template_obj)
            else:
                template_obj.is_square = False
                self.non_square_templates.append(template_obj)
        self.all_templates = self.square_templates + self.non_square_templates
        
    def load_grid(self):
        """
        嘗試從 grid_path 中載入先前的 BaseGrid 設定 (pickle 檔)，
        若存在則將其指定給 self.grid。
        """
        if self.grid_path.exists():
            self.grid = BaseGrid.load(str(self.grid_path), self.window_size)
    
    @staticmethod
    def load_config(config_file: Path):
        """
        讀取外部 JSON 設定檔，並以 dict 形式回傳。
        """
        with config_file.open("r") as file:
            return json.load(file)
    
    def search_cell_size(self, img):
        """
        嘗試透過 all_templates 與整張影像的比對，找出某個 template 的最佳匹配縮放，
        以此推估後續 cell size 的大小；主要用於除錯或自動偵測。
        """
        for template_obj in self.all_templates:
            template_shape = template_obj.img.shape
            print(f'Processing {template_obj.name} with shape: {template_shape}')
            match_scale, match_score, _ = template_matching(
                template=template_obj.img, 
                roi=img,
                iou_threshold=self.grid_matching_params["iou_threshold"],
                scale_range=self.grid_matching_params["scale_range"],
                scale_step=self.grid_matching_params["scale_step"],
                threshold=self.grid_matching_params["threshold"],
                min_area=self.grid_matching_params["min_area"],
                border=0,
                grayscale=self.initialize_grid_mode == "gray"
            )
            if match_scale is not None:
                self.estimated_cell_size = (template_shape[0] * match_scale,
                                            template_shape[1] * match_scale)
                print(f'best scale: {match_scale}, score: {match_score}, estimated cell size: {self.estimated_cell_size}')
                break
    
    def resize_templates_by_cell_size(self):
        """
        依照 self.cell_size 去縮放載入的模板圖，
        若是方形符號，直接按照 cell_height 做等比例縮放；
        若是非方形符號，額外乘上 non_square_scale。
        """
        for template_obj in self.all_templates:
            cell_width = self.cell_size[0]
            cell_height = self.cell_size[1]
            template_shape = template_obj.img.shape

            if template_obj.is_square:
                scale = cell_height / template_shape[0]
                template_obj.img = cv2.resize(template_obj.img, (0, 0), fx=scale, fy=scale)
            else:
                scale = cell_height / template_shape[0] * self.non_square_scale
                template_obj.img = cv2.resize(template_obj.img, (0, 0), fx=scale, fy=scale)

            if self.debug:
                print(f'{template_obj.name} shape: {template_obj.img.shape}')
        
    def initialize_grid(self, img):
        """
        初始化盤面：
        1. 若已經有 grid 且其形狀已穩定，且 use_saved_grid 為 True，則跳過。
        2. 否則會以 grid_matching_params 和所有樣板去匹配畫面，
           找出每個符號中心位置，並透過 get_grid_info() 建立 BaseGrid。
        3. 將建好的 grid 存成 pkl 檔，若 debug 模式則可視覺化顯示辨識結果。
        """
        start_time = time.time()
        if self.grid is not None and self.grid_is_stabilized and self.use_saved_grid:
            return
        print("Initializing grid")
        old_grid = None
        if self.grid is not None:
            old_grid = self.grid

        # 擷取中間區域（排除邊界），避免包含過多 UI
        grid_border = self.grid_matching_params["border"]
        roi = img[grid_border: -grid_border, grid_border: -grid_border]
        
        if self.debug:
            cv2.imshow("roi", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        matched_positions = process_template_matches(
            template_list=self.all_templates,
            roi=roi,
            **self.grid_matching_params,
            grayscale=self.initialize_grid_mode == "gray" or
                      (self.initialize_grid_mode == "freegame_gray" and self.mode == "free"),
            debug=self.debug
        )

        if not matched_positions:
            raise ValueError("Could not find any matches")
        
        # 根據匹配到的座標，推算出盤面 bbox 與 row/col
        grid_bbox, grid_shape = get_grid_info(matched_positions)
        self.grid = BaseGrid(grid_bbox, grid_shape, self.window_size)

        # 若目錄不存在，則先建立
        if not self.grid_path.parent.exists():
            self.grid_path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_saved_grid:
            self.grid.save(str(self.grid_path))
        
        if self.debug:
            print(f'found {len(matched_positions)} matches')
            print(f'Grid shape {self.grid.row} x {self.grid.col}')
            print(f'cell size: {self.grid.symbol_width} x {self.grid.symbol_height}')
            img = draw_grid_on_image(img, self.grid)

        elapsed_time = time.time() - start_time
        print(f"Time taken to initialize grid: {elapsed_time:.2f} seconds")

        # 若之前已存在 grid，且新舊 grid 形狀相同，則視為已穩定
        if old_grid is not None and self.use_saved_grid:
            if old_grid.col == self.grid.col and old_grid.row == self.grid.row:
                self.grid_is_stabilized = True
                print("The grid is stabilized")
            else:
                print("The grid is not stabilized! Try to modify the config")

        if self.debug:
            for pos in matched_positions:
                img = cv2.circle(img, (int(pos[0]), int(pos[1])), 5, (0, 0, 255), -1)
            for template_obj in self.all_templates:
                if template_obj.best_scale is not None:
                    print(f'{template_obj.name} best_scale: {template_obj.best_scale:.3f} '
                          f'score: {template_obj.match_score:.3f}')
                else:
                    print(f'{template_obj.name} best_scale: None')
            print()
            cv2.imshow("grid", img)
            cv2.imshow("roi", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # 在每次初始化盤面後，重置所有模板的 best_scale
        for template_obj in self.all_templates:
            template_obj.best_scale = None
        
    def recognize_roi(self, img):
        """
        對 grid 內每個 cell 進行符號辨識：
        1. 先清空現有的 _grid 資料
        2. 根據 cell_matching_method 的不同，先嘗試 template matching 或 SIFT
           若失敗則 fallback 到另一種方式
        3. 將結果 (symbol, score) 填入到 grid[i, j]
        """
        if self.grid is None:
            raise ValueError("Grid has not been initialized")
        self.grid.clear()

        for j in range(self.grid.col):
            for i in range(self.grid.row):
                if self.debug:
                    print(f"Processing roi({i}, {j})")
                # 取得該 cell 在畫面上的座標範圍
                roi = self.grid.get_roi(i, j)
                x, y, w, h = roi
                x1, x2 = x - self.cell_border, x + w + self.cell_border
                y1, y2 = y - self.cell_border, y + h + self.cell_border
                cell_with_border = img[y1:y2, x1:x2]
                cell = img[y:y+h, x:x+w]
                
                # 根據三種辨識模式決定優先策略
                if self.cell_matching_method == 0:
                    """ template only """
                    matched_obj, score = process_template_matches(
                        template_list=self.all_templates,
                        roi=cell_with_border,
                        **self.cell_matching_params,
                        grayscale=self.cell_matching_mode == "gray" or
                                  (self.cell_matching_mode == "freegame_gray" and self.mode == "free"),
                        debug=self.debug
                    )
                
                elif self.cell_matching_method == 1:
                    """ template first """
                    matched_obj, score = process_template_matches(
                        template_list=self.all_templates,
                        roi=cell_with_border,
                        **self.cell_matching_params,
                        grayscale=self.cell_matching_mode == "gray" or
                                  (self.cell_matching_mode == "freegame_gray" and self.mode == "free"),
                        debug=self.debug
                    )
                    # 若模板匹配失敗，則改用 SIFT
                    if matched_obj is None:
                        if self.debug:
                            print(f'template matching failed, using SIFT')
                        matched_obj, score = process_template_matches_sift(
                            template_list=self.all_templates,
                            roi=cell,
                            **self.sift_matching_params,
                            debug=self.debug
                        )
                        if self.debug and matched_obj is not None:
                            print(f"roi({i}, {j}) symbol: {matched_obj.name}, score: {score}")
                            cv2.imshow("cell", cell)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                
                elif self.cell_matching_method == 2:
                    """ SIFT first """
                    matched_obj, score = process_template_matches_sift(
                        template_list=self.all_templates,
                        roi=cell,
                        **self.sift_matching_params,
                        debug=self.debug
                    )
                    # 若 SIFT 失敗，再回頭用 template matching
                    if matched_obj is None:
                        if self.debug:
                            print(f'SIFT failed, using template matching')
                        matched_obj, score = process_template_matches(
                            template_list=self.all_templates,
                            roi=cell_with_border,
                            **self.cell_matching_params,
                            grayscale=self.cell_matching_mode == "gray" or
                                      (self.cell_matching_mode == "freegame_gray" and self.mode == "free"),
                            debug=self.debug
                        )
                        if self.debug and matched_obj is not None:
                            print(f"roi({i}, {j}) symbol: {matched_obj.name}, score: {score}")
                            cv2.imshow("cell", cell)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                # 如果還是 None，代表未匹配到任何符號
                if matched_obj is None:
                    if self.debug:
                        cv2.imshow("cell", cell)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    self.grid[i, j] = {"symbol": None, "score": None, "value": None}
                    continue
                
                # 將辨識到的結果存入 grid
                self.grid[i, j] = {"symbol": matched_obj.name, "score": score, "value": None}
        
    def save_grid_results(self, file_name):
        """
        將當前 grid 內的辨識結果以 JSON 檔方式輸出，
        檔名為 file_name.json。
        """
        self.grid.save_results_as_json(save_dir=self.output_json_dir,
                                       template_dir=self.template_dir,
                                       file_name=file_name)
        
    def save_annotated_frame(self, img, file_name):
        """
        將當前 grid 內的辨識結果繪製到影像上，並輸出為一張帶有符號標註的圖片。
        """
        save_path = self.save_dir / f"{file_name}.png"
        draw_bboxes_and_icons_on_image(img, self.template_dir, self.grid, save_path=save_path)

class BullGridRecognizer(BaseGridRecognizer):
    """
    BullGridRecognizer 繼承自 BaseGridRecognizer，
    並使用 BullGrid 來管理盤面，可以動態增減 row，
    同時在辨識完每列符號後檢查箭頭符號，以更新 column_heights。
    """
    def __init__(self, game: str, mode: str, config_file: Path, window_size=(1920, 1080), debug=False):
        super().__init__(game, mode, config_file, window_size, debug)
        self.is_init_column_heights = False
        # 若為 base 模式則方向為 'up'，若為 free 模式則方向為 'down'
        self.direction = 'up' if self.mode == 'base' else 'down'
        # arrow_scale_range：用來匹配上下箭頭 (up_arrow / down_arrow) 的縮放範圍
        self.arrow_scale_range = self.config["arrow_scale_range"]
        
    def load_templates(self):
        """
        與父類似，但額外載入箭頭模板 (up_arrow 或 down_arrow)，
        若找不到對應檔案則拋出錯誤。
        """
        super().load_templates()
        arrow_name = "up_arrow" if self.mode == 'base' else "down_arrow"
        for template_obj in self.all_templates:
            if template_obj.name == arrow_name:
                self.arrow_template_obj = template_obj
                break
        if self.arrow_template_obj is None:
            raise ValueError(f"Arrow template '{arrow_name}' not found in templates.")
        
    def initialize_grid(self, img):
        """
        與 BaseGridRecognizer 初始化類似，不過結束後會將 self.grid
        替換為 BullGrid，使其具備增長 row 的功能。
        """
        if self.grid is not None:
            return
        super().initialize_grid(img)
        growth_direction = 'up' if self.mode == 'base' else 'down'
        self.grid = BullGrid(
            bbox=self.grid.bbox,
            shape=(self.grid.row, self.grid.col),
            growth_direction=growth_direction
        )
            
    def recognize_roi(self, img, method):
        """
        針對 BullGridRecognizer，除了基本的辨識流程外，還會在最後檢查箭頭符號，
        若有箭頭，將會更新對應欄位的高度，必要時也會動態增加 row。
        
        method: int
            0: 全程使用 template matching
            1: template matching (灰階)
            2: 使用 SIFT
        """
        if self.grid is None:
            raise ValueError("Grid has not been initialized")
        self.grid.clear()
        
        # 依照 'base' 或 'free' 不同，取得要遍歷的 row 範圍
        for j in range(self.grid.col):
            if self.mode == 'base':
                row_range = range(self.grid.row - self.grid.column_heights[j], self.grid.row)
            else:  # self.mode == 'free'
                row_range = range(self.grid.column_heights[j])
            
            for i in row_range:
                roi = self.grid.get_roi(i, j)
                x, y, w, h = roi
                x1, x2 = x - self.cell_border, x + w + self.cell_border
                y1, y2 = y - self.cell_border, y + h + self.cell_border
                cell_with_border = img[y1:y2, x1:x2]
                
                matched_obj, score = process_template_matches(
                    template_list=self.all_templates,
                    roi=cell_with_border,
                    **self.cell_matching_params,
                    debug=self.debug
                )
                
                if matched_obj is None:
                    self.grid[i, j] = None
                    continue
                self.grid[i, j] = {"symbol": matched_obj.name, "score": score, "value": None}
        
        # 如果尚未初始化 column_heights，就做一次
        if not self.is_init_column_heights:
            self.grid.init_column_heights()
            self.is_init_column_heights = True
            print("initial column heights:", self.grid.column_heights)
        
        # 在每一欄中檢查箭頭符號是否出現，如果出現就更新高度
        for j in range(self.grid.col):
            if self.mode == 'base':
                index = self.grid.row - self.grid.column_heights[j] - 1
                position = 'top'
            else:  # self.mode == 'free'
                index = self.grid.column_heights[j]
                position = 'bottom'

            params = self.cell_matching_params.copy()
            params['scale_range'] = self.arrow_scale_range
            roi = self.grid.get_roi(index, j)
            x, y, w, h = roi
            cell = img[y:y+h, x:x+w]
            matched_obj, score = process_template_matches(
                template_list=[self.arrow_template_obj],
                roi=cell,
                **params,
                debug=self.debug
            )
            
            if matched_obj is not None:
                # 若偵測到箭頭，更新該欄位的高度
                self.grid.column_heights[j] += 1
                if self.grid.column_heights[j] > self.grid.max_height:
                    self.grid.column_heights[j] = self.grid.base_height
                print(f'{self.direction} arrow found at column[{j}], update column_heights[{j}] to {self.grid.column_heights[j]}')

                # 若新的高度超過目前 row 的數量，就新增一 row
                if self.grid.column_heights[j] > self.grid.row:
                    self.grid.add_row(position=position)
                    print(f'Updated grid shape: {self.grid.row} x {self.grid.col}')
                
        print("column heights:", self.grid.column_heights)
