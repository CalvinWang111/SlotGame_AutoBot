import os
import json
import re
import pandas as pd
from collections import defaultdict

def parse_filename(filename):
    match = re.match(r"(.+)_round_(\d+)-", filename)
    if match:
        game_name, round_number = match.groups()
        return game_name, int(round_number)
    return None, None

def extract_value(value_str):
    if not value_str:
        return 0
    return int(value_str.replace(',', '').replace('.', ''))

def compute_rtp(data_folder, output_excel=r"C:\Users\13514\SlotGame_AutoBot\excel\rtp_results.xlsx"):
    rounds_data = defaultdict(lambda: {'start_balance': None, 'end_balance': None, 'bet': None, 'win': 0})
    
    for file in sorted(os.listdir(data_folder)):
        if file.endswith(".json"):
            game_name, round_number = parse_filename(file)
            if game_name is None:
                continue
            
            file_path = os.path.join(data_folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            balance = extract_value(data.get("玩家剩餘金額", {}).get("value", "0"))
            bet = extract_value(data.get("押注金額", {}).get("value", "0"))
            win = extract_value(data.get("玩家贏分", {}).get("value", "0"))
            
            if rounds_data[round_number]['start_balance'] is None:
                rounds_data[round_number]['start_balance'] = balance
                rounds_data[round_number]['bet'] = bet  # 只取該輪第一個押注金額
            rounds_data[round_number]['end_balance'] = balance
            rounds_data[round_number]['win'] += win

    # 轉換成 DataFrame 並排序
    results = []
    total_bet, total_win = 0, 0

    for rnd in sorted(rounds_data.keys()):  # 按 round 數排序
        data = rounds_data[rnd]
        if data['win'] == 0 and data['start_balance'] is not None and data['end_balance'] is not None:
            data['win'] = data['end_balance'] - data['start_balance']
        
        total_bet += data['bet'] if data['bet'] else 0
        total_win += data['win']

        rtp = data['win'] / data['bet'] if data['bet'] else 0
        results.append([rnd, data['bet'], data['win'], f"{rtp:.2%}"])
        print(f"Round {rnd}: Bet = {data['bet']}, Win = {data['win']}, RTP = {rtp:.2%}")
    
    overall_rtp = total_win / total_bet if total_bet else 0
    print(f"Overall RTP: {overall_rtp:.2%}")

    # 存入 Excel
    df = pd.DataFrame(results, columns=["Round", "Bet", "Win", "RTP"])
    df.to_excel(output_excel, index=False)
    print(f"RTP 結果已儲存至 {output_excel}")

    return overall_rtp

# 執行程式
data_folder = r'C:\Users\13514\SlotGame_AutoBot\output\dragon\numerical'
compute_rtp(data_folder)

