import pickle
from pathlib import Path
import json
from typing import Tuple

# BaseGrid 類別用於定義最基本的盤面 (Grid) 結構與操作方法
class BaseGrid:
    def __init__(self, bbox, shape, window_size=(1920, 1080)):
        # self.window_size：當前遊戲畫面的寬高，用於紀錄網格初始化的環境大小
        self.window_size = window_size

        # self.bbox：此網格在遊戲畫面中的外框 (x, y, w, h)，其中 (x, y) 為左上角座標，w 與 h 為寬高
        self.bbox = bbox

        # self.row, self.col：網格的總列數與總行數 (shape = (row, col))
        self.row, self.col = shape

        # 依照 bbox 計算出每個符號 (cell) 在畫面上所佔的寬與高
        x, y, w, h = bbox
        self.symbol_width = w // self.col
        self.symbol_height = h // self.row

        # self._grid：用來儲存每個 cell 的辨識結果（通常以二維陣列形式存放）
        # 預設皆為 None，後續會存放如 {"symbol": ..., "score": ...} 等資訊
        self._grid = [[None for _ in range(self.col)] for _ in range(self.row)]
        
    def get_roi(self, i, j):
        """
        回傳第 i 列、第 j 行的 ROI（Region of Interest）於整張畫面的座標與大小
        (x, y, w, h)。
        """
        x, y, w, h = self.bbox
        return (int(x + self.symbol_width * j),
                int(y + self.symbol_height * i),
                int(self.symbol_width),
                int(self.symbol_height))
    
    def clear(self):
        """
        將整個 _grid 內容全部重設為 None，
        一般在開始新的辨識流程前可先呼叫這個方法清空先前的辨識結果。
        """
        for i in range(self.row):
            for j in range(self.col):
                self._grid[i][j] = None
    
    def load(path: str, window_size: Tuple[int, int]):
        """
        從指定的 pickle 檔案載入先前儲存的網格 (BaseGrid) 物件。
        若載入後的 grid 之 window_size 與目前的 window_size 不同，
        則會依照比例重新縮放 bbox 與 symbol_width / symbol_height。
        """
        with open(path, 'rb') as f:
            grid: BaseGrid = pickle.load(f)
            # 若先前儲存的網格之視窗大小與當前不同，需要依比例縮放
            if grid.window_size != window_size:
                print(f"Resizing grid from {grid.window_size} to {window_size}")
                scale = window_size[0] / grid.window_size[0]
                grid.window_size = window_size
                grid.bbox = (
                    int(grid.bbox[0] * scale),
                    int(grid.bbox[1] * scale),
                    int(grid.bbox[2] * scale),
                    int(grid.bbox[3] * scale)
                )
                grid.symbol_width = grid.bbox[2] // grid.col
                grid.symbol_height = grid.bbox[3] // grid.row
            return grid
        
    def save(self, path: str):
        """
        將目前的網格 (BaseGrid) 物件以 pickle 方式序列化並儲存到指定路徑。
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def save_results_as_json(self, save_dir: Path, template_dir: Path, file_name: str):
        """
        將網格中的辨識結果 (symbol、score、ROI 等資訊) 輸出為 JSON 檔，
        方便後續整合或做紀錄。
        """
        output_list = []
        for i in range(self.row):
            for j in range(self.col):
                cell = self._grid[i][j]
                # 若該 cell 為 None 或沒有辨識到 symbol，則略過
                if cell is None or cell["symbol"] is None:
                    continue

                # 彙整該 cell 的輸出資訊
                output_dict = {
                    "key": cell["symbol"],
                    "path": str(template_dir / f'{cell["symbol"]}.png'),
                    "confidence": float(cell["score"]),
                    "contour": self.get_roi(i, j),
                    "value": [i, j]
                }
                output_list.append(output_dict)
        
        # 將最終整理好的結果輸出為 JSON 檔案
        with open(str(save_dir / f"{file_name}.json"), "w") as f:
            json.dump(output_list, f, indent=4)
    
    def __getitem__(self, idx):
        """
        使得 grid[i, j] 可以直接取得該位置的資料，
        等同於 self._grid[i][j]。
        """
        i, j = idx
        return self._grid[i][j]

    def __setitem__(self, idx, value):
        """
        使得 grid[i, j] 可以直接被指派資料，
        等同於 self._grid[i][j] = value。
        """
        i, j = idx
        self._grid[i][j] = value
    
# BullGrid 類別為 BaseGrid 的延伸，
# 主要在於可紀錄並動態修改欄位高度 (column_heights)，
# 並可依據某些遊戲機制向上或向下增長 row。
class BullGrid(BaseGrid):
    def __init__(self, bbox, shape, growth_direction='up'):
        # 呼叫父類別 (BaseGrid) 的初始化方法
        super().__init__(bbox, shape)
        # growth_direction：紀錄此時可能的成長方向 ('up' 或 'down')
        self.growth_direction = growth_direction

        # column_heights：用來追蹤各行的當前高度，預設為整個 row 的大小
        self.column_heights = [self.row] * self.col

        # base_height：可定義最初始的高度
        self.base_height = 3
        # max_height：可定義此盤面最大容許的高度
        self.max_height = 7
    
    def init_column_heights(self):
        """
        依據設定的成長方向 ('up' 或 'down')，
        初步掃描已有的 _grid 內容，來更新各 column 的高度。
        """
        if self.growth_direction == 'up':
            # 若為 'up'，則從上往下檢查，找到第一個非 None 的位置，
            # 並據此推算該行在上方已使用的行數
            for j in range(self.col):
                for i in range(self.row):
                    if self._grid[i][j] is not None:
                        self.column_heights[j] = self.row - i
                        break
        elif self.growth_direction == 'down':
            # 若為 'down'，則從下往上檢查
            for j in range(self.col):
                for i in range(self.row - 1, -1, -1):
                    if self._grid[i][j] is not None:
                        self.column_heights[j] = i + 1
                        break

    def add_row(self, position='top'):
        """
        動態新增一列到盤面的最上方或最下方，
        同時調整 bbox 與 _grid 的結構。
        position 可為 'top' 或 'bottom'。
        """
        x, y, w, h = self.bbox

        # row 數量 +1
        self.row += 1

        # 根據新增的位置來調整 bbox (y, h) 與 _grid 的索引
        if position == 'top':
            # 若在頂端新增 row，需要往上擴大 bbox
            y -= self.symbol_height
            h += self.symbol_height
            self.bbox = (x, y, w, h)

            # 在 _grid 的最前面插入新的列 (全部為 None)
            self._grid.insert(0, [None for _ in range(self.col)])
        elif position == 'bottom':
            # 若在底部新增 row
            h += self.symbol_height
            self.bbox = (x, y, w, h)

            # 在 _grid 的最後面追加新的列
            self._grid.append([None for _ in range(self.col)])
        else:
            raise ValueError("position must be 'top' or 'bottom'")
