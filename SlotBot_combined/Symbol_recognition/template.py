import numpy as np
from pathlib import Path
import cv2

# Template 類別用於封裝符號（Symbol）相關的圖像與屬性資訊
class Template:
    def __init__(self, path: Path, img: np.ndarray):
        # self.path：儲存此樣板圖檔的完整路徑
        self.path = path
        
        # self.name：從檔案路徑中取得的檔名（不含副檔名），做為此樣板的識別名稱
        self.name = path.stem
        
        # self.img：此樣板對應的影像陣列
        self.img = img
        
        # self.best_scale：在實際辨識過程中，若找出最佳縮放比，會存於此屬性
        # 初始為 None，表示尚未計算或尚無匹配結果
        self.best_scale = None
        
        # self.is_square：用於判斷此樣板是否為方形
        # 預設為 False，通常會在程式載入模板後再行設定
        self.is_square = False
        
        # self.match_score：儲存此樣板與影像進行模板匹配後得到的最高匹配分數
        self.match_score = 0