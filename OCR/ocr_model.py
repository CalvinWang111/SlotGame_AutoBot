"""
Script for class COR
"""
import cv2
from paddleocr import PaddleOCR

class OCRmodel(PaddleOCR):
    """
    class of OCR
    """
    def __init__(self, use_angle_cls=True, lang='ch'):
        super().__init__(use_angle_cls=use_angle_cls, lang=lang)
        self.__free_game_patterns = ['免費', '免费', ]
        self.__free_game_btn = ['步步高', '免費旋轉', '旋轉']
        self.__receive_btn = "領取"
        self.__confirm_btn = "確認"
        self.__start_spin_loc = []

    def __is_free_game_patterns(self, string) -> bool:
        """
        matching free game patterns
        """
        for pattern in self.__free_game_patterns:
            if pattern in string:
                return True
        return False

    def __get_free_game_loc(self, loc_list) -> list[tuple]:
        """
        get int type location
        """

        return [(int(loc[0]), int(loc[1])) for loc in loc_list]


    def is_freegame(self, response_list) -> bool:
        """
        Determine whether it enters the free game or the base game
        """

    def get_free_game_btn(self, image_path, result_path) -> list:
        """
        Get free game button list
        loc: [l_up, r_up, r_down, l_down]
        """
        image = cv2.imread(image_path)
        free_game_btn_loc = list()
        responses = self.ocr(image_path)[0]
        for response in responses:
            print(response[1][0])
            if self.__is_free_game_patterns(response[1][0]):
                loc = self.__get_free_game_loc(response[0])
                free_game_btn_loc.append(loc)
                x1, y1 = loc[0][0], loc[0][1]
                x2, y2 = loc[2][0], loc[2][1]
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2, 2)

        cv2.imwrite(result_path, image)
