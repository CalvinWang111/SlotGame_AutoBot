# SlotGame_AutoBot
# 自動化轉輪機器人 產學案

**可用branches已經依照各組別名稱創立**
可依照需求自行放上文件及程式碼

**遊戲素材蒐集**
原則以模擬器截取橫版/直版，1920*1080
資料夾'material'依需求放入分割完元件或是遊戲畫面截圖

## ./VisionTransformer VIT模型訓練模組
```
VIT_Trainer 訓練模組，已經預設最佳參數
Train       呼叫模組進行訓練
```
```
SlotBot folder: 模組化操作元件辨識，須放在SAM/notebooks 路徑底下，VIT模型檔下載路徑如下
使用Bluestacks模擬器執行遊戲(不能覆蓋其他視窗在上面)，運行main.py即可執行，
元件指令如下
    # 4. 操控遊戲
    GameController.Windowcontrol(GameController,highest_confidence_images=highest_confidence_images, classId=8)
更換classId 使用底下編號與按鍵對應表
label_map = {
    0: "button_max_bet",
    1: "button_additional_bet",
    2: "button_close",
    3: "button_decrease_bet",
    4: "button_home",
    5: "button_increase_bet",
    6: "button_info",
    7: "button_speedup_spin",
    8: "button_start_spin",
    9: "button_three_dot",
    10: "gold_coin",
    11: "gold_ingot",
    12: "stickers"
}

vit_model3.pth  模型檔(沒有free game圖資) / VITrun_ver6(含有free game圖資訓練)
https://drive.google.com/drive/folders/1om612fqjcpM44GijEISI0M3zDaYmeD4n?usp=sharing
```
