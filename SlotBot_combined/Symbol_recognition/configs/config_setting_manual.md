## config設定項目說明
#### 路徑相關參數
* `template_dir`, `save_dir`, `grid_path`, `output_json_dir` - 原則上不需調整

#### 盤面辨識微調參數
* `cell_border`
* `grid_matching_params` - 設定初始定位時的相關參數。
  * `iou_threshold`
  * `scale_range` - 符號縮放大小的上下限，若有辨識不到或誤判背景的情形，可以依照debug模式的輸出進行調整。
  * `scale_step` -  符號調整縮放時的單位，設定越小耗時越久，不過精確度可能更高。
  * `threshold` - 辨識結果分數的下限，若發生誤判時可以嘗試調高，若辨識不到則調低，是對結果有很大影響的重要參數。此外，若將 **"initialize_grid_mode"** 由 **"color"** 更改為 **"gray"** 時，應該適度**調高**此參數，反之亦然。
  * `min_area`
  * `border`

* `cell_matching_params` - 設定使用template matching時的相關參數。由於盤面定位也是採用template matching的方式，許多參數和設定原則是相同的。
  * `iou_threshold`
  * `scale_range` - 符號縮放大小的上下限，建議範圍可以略高於或等於grid_matching_params中的設定。
  * `scale_step`
  * `threshold` - 將 **"cell_matching_mode"** 由 **"color"** 更改為 **"gray"** 時，應該適度**調高**此參數，反之亦然。
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

  分別代表初始化盤面以及辨識每格內容時是否將圖片轉為灰階，可選參數有 **"color"**, **"gray"** ,以及 **"freegame_gray"**
  使用gray時，效率會較高，但也更可能造成誤判，需要設定更高的treshhold。
  針對banana tower等有「同形異色」且意義不同的symbol的遊戲，應設為 **"color"**；而像金猴爺和福星高照的free game等希望在free game時用原有symbol辨識不同顏色的版本時，則應該設為 **"freegame_gray"**。

`cell_matching_method` (可選，預設為 **"sift first"**)

  可選參數有 **"template only"**, **"template first"**, **"sift first"**，分別對應「只用template matching」、「先用template matching，若失敗再用sift」，以及「先用sift，若失敗再用template matching」。針對banana tower等有「同形異色」且意義不同的symbol的遊戲，應設為 **"template only"** 或 **"template first"**。

* `resize_template` (可選，預設為 **true**)

  若設為true，將不會儲存符號的大小，用於因應符號大小會變動的場合(例如可變盤面)。

* `use_saved_grid` (可選，預設為 **true**)
  
  若設為true，將不會儲存和讀取grid.pkl，而是於每張畫面重新進行初始盤面定位。用於因應盤面位置在同一種模式(base/free)下也會變動的場合。

#### 停輪辨識微調參數
* `timing_offset` (可選，預設為 **0**) - 調整捕捉key_frame的時機，以幀為單位(可以用30fps計算)，建議最小值為-5，且不得小於-8，因為負值會改變緩衝區的大小進而影響精確度。如果感覺程式捕捉key_frame的時機太早，可以嘗試將其調大，例如在停止時有較長漸慢時間的競品，或者吉祥如意的free game等會在旋轉停止後還有額外動畫的場合，便可調整此參數。

* `feature_sensitivity` (可選，預設為 **1**) - 調整角點偵測的靈敏度，有效範圍為(0,100]。如果遊戲中有對比度不高，或者輪廓不清晰的符號(例如低對比的灰階符號)，增加此值可以增加該符號的辨識率，但也會花費較多時間且更容易受雜訊干擾。(但通常情況下，問題不會在這裡)

* `optical_flow_fineness` (可選，預設為 **1**) - 調整光流偵測的精細度，標準值為1。設定得較高時，會捕捉到更精確的光流，但也會花費較多時間。橫版遊戲建議設定範圍為(0,5]，否則可能導致fps不如預期。

* `moving_distance_rate` (可選，預設為 **1**) - 設定「需要多長的光流才會被視為轉動訊號」。可以依據遊戲轉輪的速度進行微調。

* `use_upward_flow` (可選，預設為 **false**) - 在某些情況下，程式難以在轉輪轉動時捕捉到方向向下的光流，反而會捕捉到方向向上的，也許這就跟高速轉動的輪胎看起來像倒著轉的原理差不多吧。如果常常發生轉輪轉到一半就被判定停輪的情況可以嘗試開啟，但也會相對地增加連線特效被誤判的機會。

* `optical_flow_strict_mode` (可選，預設為 **false**) - 非垂直光流會抵消垂直光流的數量，能夠有效減少連線特效干擾的情形。建議將flow_fineness設定調高時再開啟；在盤面範圍內會有持續動態(例如邊框特效)時不建議開啟。

#### 停輪辨識其他參數
* `use_key_frame` (可選，預設為 **false**)

  若設為true，將在base game中也使用key frame進行盤面和數值辨識，主要用於因應龍騎士的特殊規則。在free game狀態下不會有影響。


## 其他注意事項
若初始定位得到了錯誤的盤面結果，請先將../grids中對應的檔案刪除，之後再進行其他嘗試，否則程式只會一直讀取原本錯誤的盤面資訊，而不會重新定位
