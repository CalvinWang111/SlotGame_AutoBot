import cv2
import numpy as np
from pathlib import Path
import json
from .grid import *
from .symbol_recognizer import *
from .utils import *

class BaseGridRecognizer:
    def __init__(self, game:str, mode:str, config_file: Path, window_size=(1920, 1080), debug=False):
        self.game = game
        self.mode = mode
        self.config = self.load_config(config_file)
        self.template_dir = Path(self.config["template_dir"].format(game=game, mode=mode))
        self.save_dir = Path(self.config["save_dir"].format(game=game, mode=mode))
        self.grid_path = Path(self.config["grid_path"].format(game=game, mode=mode))
        self.output_json_dir = Path(self.config["output_json_dir"].format(game=game, mode=mode))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.output_json_dir.mkdir(parents=True, exist_ok=True)
        
        # Define window size and adjustment ratio
        self.layout_orientation = self.config["layout_orientation"]
        horizontal_base_size = (1920, 1080)
        vertical_base_size = (1080, 1920)
        if self.layout_orientation == "horizontal":
            self.adjustment_ratio = window_size[0] / horizontal_base_size[0]
        elif self.layout_orientation == "vertical":
            self.adjustment_ratio = window_size[1] / vertical_base_size[1]
        print(f'Adjustment ratio: {self.adjustment_ratio}')
        # Multiply the adjustment ratio to the grid matching parameters
        self.cell_border = int(self.config["cell_border"] * self.adjustment_ratio)
        self.grid_matching_params = self.config["grid_matching_params"]
        self.grid_matching_params['scale_range'][0] *= self.adjustment_ratio
        self.grid_matching_params['scale_range'][1] *= self.adjustment_ratio
        self.grid_matching_params['scale_step'] *= self.adjustment_ratio
        self.grid_matching_params['min_area'] = int(self.grid_matching_params['min_area'] * self.adjustment_ratio)
        self.grid_matching_params['border'] = int(self.grid_matching_params['border'] * self.adjustment_ratio)
        self.cell_matching_params = self.config["cell_matching_params"]
        self.cell_matching_params['scale_range'][0] *= self.adjustment_ratio
        self.cell_matching_params['scale_range'][1] *= self.adjustment_ratio
        self.cell_matching_params['scale_step'] *= self.adjustment_ratio
        self.cell_matching_params['min_area'] = int(self.cell_matching_params['min_area'] * self.adjustment_ratio)
        self.cell_matching_params['border'] = int(self.cell_matching_params['border'] * self.adjustment_ratio)
        self.sift_matching_params = self.config["sift_matching_params"]
        
        self.initialize_grid_mode = self.config["initialize_grid_mode"]
        
        self.grid = None
        self.load_grid()
        self.template_match_data = {}
        self.debug = debug
        
    def load_grid(self):
        if self.grid_path.exists():
            self.grid = BaseGrid.load(str(self.grid_path))
    
    @staticmethod
    def load_config(config_file: Path):
        with config_file.open("r") as file:
            return json.load(file)
        
    def initialize_grid(self, img):
        if self.grid is not None:
            return
        print("Initializing grid")
        use_gray = self.initialize_grid_mode == "gray"
        process_template_matches(
            template_match_data=self.template_match_data,
            template_dir=self.template_dir,
            img=img,
            **self.grid_matching_params,
            grayscale=use_gray
        )

        matched_positions = [
            (top_left[0] + data["shape"][1] * scale / 2,
             top_left[1] + data["shape"][0] * scale / 2)
            for template_name, data in self.template_match_data.items()
            for (top_left, scale, _) in data["result"]
        ]
        if not matched_positions:
            raise ValueError("Could not find any matches")
        
        grid_bbox, grid_shape = get_grid_info(matched_positions)
        self.grid = BaseGrid(grid_bbox, grid_shape)
        if not self.grid_path.parent.exists():
            self.grid_path.parent.mkdir(parents=True, exist_ok=True)
        self.grid.save(str(self.grid_path))
        
        if self.debug:
            print(f'Grid shape {self.grid.row} x {self.grid.col}')

    def recognize_roi(self, img, method):
        """
        method: int
            0: use template matching
            1: use template mathhing gray
            2: use SIFT
        """
        if self.grid is None:
            raise ValueError("Grid has not been initialized")
        self.grid.clear()

        for j in range(self.grid.col):
            for i in range(self.grid.row):
                if self.debug:
                    print(f"Processing roi({i}, {j})")
                roi = self.grid.get_roi(i, j)
                x, y, w, h = roi
                x1, x2 = x - self.cell_border, x + w + self.cell_border
                y1, y2 = y - self.cell_border, y + h + self.cell_border
                cell_with_border = img[y1:y2, x1:x2]
                cell = img[y:y+h, x:x+w]

                if method == 0:
                    symbol_name, score = process_template_matches(
                        template_match_data=self.template_match_data,
                        template_dir=self.template_dir,
                        img=cell_with_border,
                        **self.cell_matching_params,
                        debug=self.debug
                    )
                elif method == 1:
                    symbol_name, score = process_template_matches(
                        template_match_data=self.template_match_data,
                        template_dir=self.template_dir,
                        img=cell_with_border,
                        **self.cell_matching_params,
                        grayscale=True,
                        debug=self.debug
                    )
                elif method == 2:
                    symbol_name, score = process_template_matches_sift(
                        template_dir=self.template_dir, 
                        target_roi=cell, 
                        **self.sift_matching_params,
                        debug=self.debug
                    )

                    # when SIFT fails, use template matching
                    if symbol_name is None:
                        symbol_name, score = process_template_matches(
                            template_match_data=self.template_match_data,
                            template_dir=self.template_dir,
                            img=cell_with_border,
                            **self.cell_matching_params,
                            debug=self.debug
                        )
                        if self.debug:
                            print(f"roi({i}, {j}) symbol: {symbol_name}, score: {score}")
                            cv2.imshow("cell", cell)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                    
                    
                
                self.grid[i, j] = {"symbol": symbol_name, "score": score, "value": None}
        
    
    def save_grid_results(self, file_name):
        self.grid.save_results_as_json(save_dir=self.output_json_dir, template_dir=self.template_dir, file_name=file_name)
        
    def save_annotated_frame(self, img, file_name):
        save_path = self.save_dir / f"{file_name}.png"
        draw_bboxes_and_icons_on_image(img, self.template_dir, self.grid, save_path=save_path)