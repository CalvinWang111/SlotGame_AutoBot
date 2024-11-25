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
        "請幫我找出畫面中的所有數值，並告訴我它的意義，輸出成 數值;意義 的格式，不要有其他符號"
        "用{}框起來，意義的部分盡量精簡"
    )

class GetSimplifiedMeaningPromptFormat:
    """
        A class for prompt format of getting value
    """
    PROMPT = (
        "我辨識了一個數字，但我不知道它的意義，以下是它可能的意義。"
        "你認為這個數字的意義是什麼。意義的部分盡量精簡。"
    )
