import json
import os
from openpyxl import Workbook
import pandas as pd
from pathlib import Path



class Excel_parser:
    def __init__(self):
        self.symbol_path = ""
        self.value_path = ""
        self.root_dir = Path(__file__).parent.parent

    def json_to_excel(self, game_type, game_state):

        Path(os.path.join(self.root_dir, "excel")).mkdir(exist_ok=True)
        save_path = os.path.join(self.root_dir, 'excel', f'{game_type}_{game_state}.xlsx')
        self.symbol_path = os.path.join(self.root_dir, 'output', game_type,'symbols')
        self.value_path = os.path.join(self.root_dir, 'output', game_type, 'numerical')
        # 建立資料
        excel = {
            "遊戲名稱": [],
            "測試時間": [],
            "遊戲狀態": [],
        }

        excel_value = {}
        excel_symbol = {}

        #讀取數值檔案
        #建立鍵值
        value_file_list = os.listdir(self.value_path)
        value_file_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for file_name in value_file_list:
            excel['遊戲名稱'].append(game_type)
            excel['測試時間'].append("")
            excel['遊戲狀態'].append(game_state)

            # 檢查是否為 JSON 檔案
            if file_name.endswith(".json"):
                file_path = os.path.join(self.value_path, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    # 解析 JSON 檔案
                    json_data = json.load(file)
                    # 建立鍵值
                    for key in json_data:
                        if not key in excel_value:
                            excel_value[key] = []

        #輸入數值
        for file_name in value_file_list:
            # 檢查是否為 JSON 檔案
            if file_name.endswith(".json"):
                file_path = os.path.join(self.value_path, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    # 解析 JSON 檔案
                    json_data = json.load(file)
                    for key in excel_value:
                        if not key in json_data:
                            excel_value[key].append("")
                        else:
                            excel_value[key].append(json_data[key]['value'])


        # 讀取盤面檔案
        # 建立鍵值
        symbol_file_list = os.listdir(self.symbol_path)
        symbol_file_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for file_name in symbol_file_list:
            # 檢查是否為 JSON 檔案
            if file_name.endswith(".json"):
                file_path = os.path.join(self.symbol_path, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    # 解析 JSON 檔案
                    json_data = json.load(file)
                    # 建立鍵值
                    for grid in json_data:
                        symbol_pos = f"C{grid['value'][0]}R{grid['value'][1]}"
                        # 建立鍵值
                        if not symbol_pos in excel_symbol:
                            excel_symbol[symbol_pos] = []

        # 輸入數值
        for file_name in symbol_file_list:
            # 檢查是否為 JSON 檔案
            if file_name.endswith(".json"):
                file_path = os.path.join(self.symbol_path, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    # 解析 JSON 檔案
                    json_data = json.load(file)

                    #problem
                    for key in excel_symbol:
                        found = False
                        for grid in json_data:
                            symbol_pos = f"C{grid['value'][0]}R{grid['value'][1]}"
                            if symbol_pos == key:
                                found = True
                                excel_symbol[key].append(grid['key'])
                                break
                        if not found:
                            excel_symbol[key].append("")


        excel.update(excel_symbol)
        excel.update(excel_value)

        # 建立 DataFrame
        df = pd.DataFrame(excel)

        # 儲存檔案
        df.to_excel(save_path)
        print("檔案已成功儲存！")
'''
if __name__ == "__main__":
    ex = Excel_parser()
    ex.json_to_excel("golden","base")
'''