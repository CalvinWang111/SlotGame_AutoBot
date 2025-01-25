"""
This script is for all prompt
"""
from dataclasses import dataclass

@dataclass
class InfoPromptFormat:
    """
    A class prompt format for asking info page
    """
    system_role_play = """你是一個slot game的遊戲管理員，你會接收到`<rule>規則`</rule>
理解圖片中slot game的規則，理解圖片上的**symbol**與其**敘述**，並判斷給定的`<game scence>遊戲畫面上的輪盤`</game scence>是否為免費遊戲。
"""
    rule_prompt = """以下圖片為`<rule>slot game的規則`</rule>以及`<base game>一般遊戲盤面`</base game>。
"""

    user_prompt = """以下圖片為`<game scence>遊戲畫面`</game scence>。
1. 請比較`<base game>一般遊戲盤面`</base game>與`<game scence>遊戲畫面`</game scence>。
2. 根據`<rule>規則`</rule>，仔細確認**symbol數量**，必須嚴格遵循遊戲規則，檢查`<game scence>遊戲畫面`</game scence>的symbol數量。
判斷畫面中的盤面是否為免費遊戲或是達到規則上可以跳離`<base game>一般遊戲盤面`</base game>。
如果**是**請輸出`<answer>是`</answer>，如果**不是**輸出`<answer>不是`</answer>。
如果無法確認，判斷為`<answer>不是`</answer>
並詳細解釋遊戲畫面出現甚麼符號或特徵符合遊戲規則，用標籤`<explain>`</explain>解釋。
"""

class GetValuePromptFormat:
    """
        A class for prompt format of getting value
    """
    PROMPT = (
        """
            我想要辨識一款slot game的遊戲畫面中的數字，並且要知道數字的意義。

            以下是slot game中主要要獲得的信息(僅供參考，實際上可能會出現其他重要信息):
                玩家贏得的分數(ex:玩家贏分)
                玩家剩餘金額(ex:玩家剩餘金額)
                每局的押注金額(ex:押注金額)
                各種獎項金額

            請幫我辨識遊戲畫面中的數字，並告訴我數字的意義，要符合以下條件:
            1.只保留重要的信息
            2.數字要用<number></number>標籤包起來，意義要用<meaning></meaning>標籤包起來
            3.輸出格式<number></number> = <meaning></meaning>
            4.回答盡量精簡
            
        """
    )

class GetSimplifiedMeaningPromptFormat:
    """
        A class for prompt format of getting value
    """
    PROMPT = (
        """
        我正在辨識一款slot game的遊戲畫面中的數字，但我得到數字的數值和數字的位置了，卻不知道它的意義，所以我記錄了這些數字的位置和這些數字可能的意義。
        請你為每一筆資料進行可信度評分，規則如下:
        1.若一個位置的意義越多，則可信度越高。
        2.若一個位置的意義越重複，則可信度越高。
        
        以下是slot game中主要要獲得的信息(僅供參考，實際上可能會出現其他重要信息):
            玩家贏得的分數(ex:玩家贏分)
            玩家剩餘金額(ex:玩家剩餘金額)
            每局的押注金額(ex:押注金額)
            各種獎項金額
        
        請幫我整理位置對上意義，並符合以下條件:
        1.只保留重要的信息
        2.若不同位置出現相同的意義請選擇可信度較高的
        3.要避免輸出重複的意義
        4.位置要用<position></position>標籤包起來，意義要用<meaning></meaning>標籤包起來
        5.輸出格式<position></position> = <meaning></meaning>
        6.回答盡量精簡
        """
    )

# """
#         我正在辨識一款slot game的遊戲畫面中的數字，但我得到數字的數值了，卻不知道它的意義，所以我記錄了這個數字可能的意義。
#         你認為這個數字的意義是什麼，遵守以下條件:
#         1.回答盡量精簡。
#         2.輸出格式<meaning></meaning>，意義要用<meaning></meaning>標籤包起來。
#         """