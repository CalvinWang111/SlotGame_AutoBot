"""
This script is for all prompt
"""
class PromptFormat:
    """
    A class for prompt format
    """
    PROMPT = (
        "我現在需要辨識輪轉遊戲畫面上的按鍵與非按鍵，"
        "遊戲畫面上的按鍵可能有<button>增加押注</button>、<button>減少押注</button>、<button>開始輪轉</button>、"
        "<button>工具列</button>、<button>關閉</button>、<button>返回主頁</button>、<button>遊戲資訊</button>、"
        "<button>快速</button>、<button>額外押注</button>、<button>最大押注</button>等按鍵。"
        "以下為按鍵的特徵，請**依照特徵辨識**："
        "<button>增加押注</button>上會有**+**的符號或是**增加**。"
        "<button>減少押注</button>上會有**-**的符號或是**減少**。"
        "<button>遊戲資訊</button>上會有字母**i**或是**info**的字"
        "<button>工具列</button>上會有**3個點**的符號。"
        "<button>返回主頁</button>上會有**房子**的符號。"
        "<button>關閉</button>上會有**X**或是**叉叉**的符號。"
        "<button>最大押注</button>上只會有**中文**字**最大押注**。"
        "<button>額外押注</button>上會有**中文**字**額外押注**。"
        "除了以上的按鍵，其餘皆當作非按鍵"
        "請幫我辨識這張圖片是不是按鍵，請在標籤`<answer></answer>`內輸出是<answer>是</answer>或者<answer>不是</answer>"
        "如果是按鍵的話，在標籤`<button></button>`內輸出是哪一類按鍵。"
    )

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
                特殊獎項金額(ex:巨獎/大獎/中獎/小獎)

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
            特殊獎項金額(ex:巨獎/大獎/中獎/小獎)
        
        請幫我整理位置對上意義，並符合以下條件:
        1.只保留重要的信息
        2.若不同位置出現相同的意義請選擇可信度較高的
        3.要避免輸出重複的意義
        4.位置要用<position></position>標籤包起來，意義要用<meaning></meaning>標籤包起來
        輸出格式<position></position> = <meaning></meaning>
        5.回答盡量精簡
        """

    )
