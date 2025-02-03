import os
import json
import re
import pandas as pd
import logging
from collections import defaultdict

# 設定 logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_filename(filename):
    """解析檔名，提取遊戲名稱與回合數"""
    match = re.search(r"(.+)_round_(\d+)-", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def extract_value(value_str):
    """提取數值，支持浮點數，並去除千分位逗號"""
    try:
        return float(value_str.replace(',', '')) if value_str else 0
    except ValueError:
        logging.warning(f"數值轉換失敗：{value_str}")
        return 0

def compute_rtp(data_folder, output_excel="rtp_results.xlsx"):
    """計算 RTP 並輸出 Excel 檔案"""
    rounds_data = {}

    if not os.path.exists(data_folder):
        logging.error(f"資料夾不存在：{data_folder}")
        return None
    
    # 遍歷 JSON 檔案，提取每個回合的數據
    for file in sorted(os.listdir(data_folder)):
        if not file.endswith(".json"):
            continue

        game_name, round_number = parse_filename(file)
        if game_name is None:
            logging.warning(f"無法解析檔名：{file}")
            continue

        file_path = os.path.join(data_folder, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"讀取失敗 {file}: {e}")
            continue

        balance = extract_value(data.get("玩家剩餘金額", {}).get("value", "0"))
        bet = extract_value(data.get("押注金額", {}).get("value", "0"))
        win = extract_value(data.get("玩家贏分", {}).get("value", "0"))  # 優先使用 JSON 提供的贏分

        if round_number not in rounds_data:
            rounds_data[round_number] = {
                "end_balance": balance, 
                "bet": bet,
                "max_win": win  # 記錄該回合的最大贏分
            }
        else:
            rounds_data[round_number]["end_balance"] = balance
            rounds_data[round_number]["max_win"] = max(rounds_data[round_number]["max_win"], win)  # 取最大值

    # 計算 RTP
    results = []
    total_bet, total_win = 0, 0
    previous_balance = None

    for rnd in sorted(rounds_data.keys()):
        data = rounds_data[rnd]

        # 計算贏分，優先使用 JSON 記錄的最大贏分
        if data["max_win"] > 0:
            win = data["max_win"]
        else:
            # 若 JSON 無提供贏分，則使用餘額變化計算
            win = max(0, data["end_balance"] - previous_balance) if previous_balance is not None else 0

        total_bet += data["bet"]
        total_win += win

        try:
            rtp = win / data["bet"] if data["bet"] > 0 else 0
        except ZeroDivisionError:
            rtp = 0

        results.append([rnd, data["bet"], win, f"{rtp:.2%}"])
        logging.info(f"Round {rnd}: Bet = {data['bet']}, Win = {win}, RTP = {rtp:.2%}")

        # 更新上一回合的餘額
        previous_balance = data["end_balance"]

    overall_rtp = total_win / total_bet if total_bet > 0 else 0
    logging.info(f"Overall RTP: {overall_rtp:.2%}")

    # 存入 Excel
    df = pd.DataFrame(results, columns=["Round", "Bet", "Win", "RTP"])
    df.to_excel(output_excel, index=False)
    logging.info(f"RTP 結果已儲存至 {output_excel}")

    return overall_rtp

# 執行程式
data_folder = r'C:\Users\13514\SlotGame_AutoBot\output\dragon\numerical'
compute_rtp(data_folder)
