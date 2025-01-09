## config設定項目說明
#### 路徑相關參數
* `template_dir`, `save_dir`, `grid_path`, `output_json_dir` - 原則上不需調整

#### 盤面辨識微調參數
* `cell_border`
* `grid_matching_params` - 設定初始定位時的相關參數。
  * `iou_threshold`
  * `scale_range` - 符號縮放大小的上下限，若有辨識不到或誤判背景的情形，可以依照debug模式的輸出進行調整。
  * `scale_step` -  符號調整縮放時的單位，設定越小耗時越久，不過精確度可能更高。
  * `threshold` - 辨識結果分數的下限，若發生誤判時可以嘗試調高，若辨識不到則調低，是對辨識結果有很大影響的重要參數。此外，若將 **"initialize_grid_mode"** 由 **"color"** 更改為 **"gray"** 時，應該適度**調高**此參數。
  * `min_area`
  * `border`

* `cell_matching_params` - 設定使用template matching時的相關參數。由於盤面定位也是採用template matching的方式，許多參數和設定原則是相同的。
  * `iou_threshold`
  * `scale_range` - 符號縮放大小的上下限，建議範圍可以略高於或等於grid_matching_params中的設定。
  * `scale_step`
  * `threshold`
  * `min_area`
  * `match_one` - 目前已棄置，但必須設定為**true**
  * `border`

* `sift_matching_params` - 設定使用sift matching時的相關參數。
  * `scale_range` - 符號縮放大小的上下限，建議範圍可以略高於或等於grid_matching_params中的設定
  * `min_matches` - 若比對到的特徵點數量高於此數量，則判定為相同的symbol。若有誤判的情形，可以嘗試調高此參數。
  * `ratio_threshold`
  * `ransac_threshold`
  * `vertical_threshold`
  * `horizontal_threshold`
* `cell_size`
* `non_square_scale`

#### 盤面辨識其他參數
* `layout_orientation`

  可選參數有 **"vertical"**, **"horizontal"**。依照模擬器的長寬比(而不是盤面本身)進行設定即可。

* `initialize_grid_mode`, `cell_matching_mode`

  分別代表初始化盤面以及辨識每格內容時是否將圖片轉為灰階，可選參數有 **"color"**, **"gray"** 
  使用gray時，效率會較高，但也更可能造成誤判，需要設定更高的treshhold。
  針對banana tower等有「同形異色」且意義不同的symbol的遊戲，應設為 **"color"**；而像金猴爺和福星高照的free game等希望用原有symbol辨識不同顏色的版本時，則應該設為 **"gray"**。

`cell_matching_method` (可選，預設為 **"sift first"**)

  可選參數有 **"template only"**, **"template first"**, **"sift first"**，分別對應「只用template matching」、「先用template matching，若失敗再用sift」，以及「先用sift，若失敗再用template matching」。針對banana tower等有「同形異色」且意義不同的symbol的遊戲，應設為 **"template only"** 或 **"template first"**。

* `resize_template` (可選，預設為 **true**)

  若設為true，將不會儲存符號的大小，用於因應符號大小會變動的場合(例如可變盤面)。

* `use_saved_grid` (可選，預設為 **true**)
  
  若設為true，將不會儲存和讀取grid.pkl，而是於每張畫面重新進行初始盤面定位。用於因應盤面位置在同一種模式(base/free)下也會變動的場合。

#### 停輪辨識微調參數
#### 停輪辨識其他參數
* `use_key_frame` (可選，預設為 **false**)

  若設為true，將在base game中也使用key frame進行盤面和數值辨識，主要用於因應龍騎士的特殊規則。在free game狀態下不會有影響。


## 其他注意事項
若初始定位得到了錯誤的盤面結果，請先將../grids中對應的檔案刪除，之後再進行其他嘗試，否則程式只會一直讀取原本錯誤的盤面資訊，而不會重新定位
