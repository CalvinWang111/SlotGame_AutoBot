import cv2
import numpy as np
from pathlib import Path
import json
import time
from .grid import *
from .template import *
from .symbol_recognizer import *
from .utils import *
from typing import List, Dict

root_dir = Path(__file__).parent.parent.parent

class BaseGridRecognizer:
    def __init__(self, game:str, mode:str, config_file: Path, window_size=(1920, 1080), debug=False):
        self.game = game
        self.mode = mode
        self.debug = debug
        self.config = self.load_config(config_file)
        self.template_dir = root_dir / Path(self.config["template_dir"].format(game=game, mode=mode))
        self.save_dir = root_dir / Path(self.config["save_dir"].format(game=game, mode=mode))
        self.grid_path = root_dir / Path(self.config["grid_path"].format(game=game, mode=mode))
        self.output_json_dir = root_dir / Path(self.config["output_json_dir"].format(game=game, mode=mode))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.output_json_dir.mkdir(parents=True, exist_ok=True)
        
        # Define window size and adjustment ratio
        self.layout_orientation = self.config["layout_orientation"]
        self.window_size = window_size
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
        self.grid_matching_params['min_area'] = int(self.grid_matching_params['min_area'] * (self.adjustment_ratio ** 2))
        self.grid_matching_params['border'] = int(self.grid_matching_params['border'] * self.adjustment_ratio)
        self.cell_matching_params = self.config["cell_matching_params"]
        self.cell_matching_params['min_area'] = int(self.cell_matching_params['min_area'] * (self.adjustment_ratio ** 2))
        self.cell_matching_params['border'] = int(self.cell_matching_params['border'] * self.adjustment_ratio)
        self.sift_matching_params = self.config["sift_matching_params"]
        
        self.initialize_grid_mode = self.config["initialize_grid_mode"]
        self.cell_matching_mode = self.config["cell_matching_mode"]

        method_list = ['template only','template first','sift first']
        cell_matching_method = self.config.get('cell_matching_method', 'sift first')
        self.cell_matching_method = method_list.index(cell_matching_method)
        
        self.cell_size = self.config["cell_size"]
        self.cell_size[0] = int(self.cell_size[0] * self.adjustment_ratio)
        self.cell_size[1] = int(self.cell_size[1] * self.adjustment_ratio)
        print(f'Cell size: {self.cell_size}')
        self.non_square_scale = self.config["non_square_scale"]
        
        self.square_templates: List[Template] = []
        self.non_square_templates: List[Template] = []
        self.all_templates: List[Template] = []
        
        self.resize_template = self.config.get('resize_template', True)
        self.load_templates()
        if self.resize_template:
            self.resize_templates_by_cell_size()
        
        self.use_saved_grid = self.config.get('use_saved_grid', True)
        self.grid = None
        if self.use_saved_grid:
            self.load_grid()
        if self.grid is None:
            self.grid_is_stabilized = False
        else:
            self.grid_is_stabilized = True
            
        self.template_match_data = {}
        self.debug = debug
        
    
    def load_templates(self):
        print(str(self.template_dir.absolute()))
        for path in self.template_dir.glob('*.png'):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            alpha_channel = img[:, :, 3]
            total_pixels = alpha_channel.size
            transparent_pixels = np.sum(alpha_channel < 32)
            transparent_ratio = transparent_pixels / total_pixels
            template_obj = Template(path, img)
            if transparent_ratio < 0.2:
                template_obj.is_square = True
                self.square_templates.append(template_obj)
            else:
                template_obj.is_square = False
                self.non_square_templates.append(template_obj)
        self.all_templates = self.square_templates + self.non_square_templates
        
    def load_grid(self):
        if self.grid_path.exists():
            self.grid = BaseGrid.load(str(self.grid_path), self.window_size)
    
    @staticmethod
    def load_config(config_file: Path):
        with config_file.open("r") as file:
            return json.load(file)
    
    def search_cell_size(self, img):
        for template_obj in self.all_templates:
            template_shape = template_obj.img.shape
            print(f'Processing {template_obj.name} with shape: {template_shape}')
            match_scale, match_score, _ = template_matching(
                template=template_obj.img, 
                roi=img,
                iou_threshold=self.grid_matching_params["iou_threshold"],
                scale_range=self.grid_matching_params["scale_range"],
                scale_step=self.grid_matching_params["scale_step"],
                threshold=self.grid_matching_params["threshold"],
                min_area=self.grid_matching_params["min_area"],
                border=0,
                grayscale=self.initialize_grid_mode=="gray"
            )
            if match_scale is not None:
                self.estimated_cell_size = (template_shape[0] * match_scale, template_shape[1] * match_scale)
                print(f'best scale: {match_scale}, score: {match_score}, estimated cell size: {self.estimated_cell_size}')
                break
    
    def resize_templates_by_cell_size(self):
        for template_obj in self.all_templates:
            # resize template base on cell size and if is a square template
            cell_width = self.cell_size[0]
            cell_height = self.cell_size[1]
            template_shape = template_obj.img.shape
            if template_obj.is_square:
                scale = cell_height / template_shape[0]
                template_obj.img = cv2.resize(template_obj.img, (0, 0), fx=scale, fy=scale)
            else:
                scale = cell_height / template_shape[0] * self.non_square_scale
                template_obj.img = cv2.resize(template_obj.img, (0, 0), fx=scale, fy=scale)
            if self.debug:
                print(f'{template_obj.name} shape: {template_obj.img.shape}')
        
    def initialize_grid(self, img):
        start_time = time.time()
        if self.grid is not None and self.grid_is_stabilized and self.use_saved_grid == True:
            return
        print("Initializing grid")
        old_grid = None
        if self.grid is not None:
            old_grid = self.grid

        grid_border = self.grid_matching_params["border"]
        roi = img[grid_border: -grid_border, grid_border: -grid_border]
        
        if self.debug:
            cv2.imshow("roi", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        matched_positions = process_template_matches(
            template_list=self.all_templates,
            roi=roi,
            **self.grid_matching_params,
            grayscale=self.initialize_grid_mode=="gray",
            debug=self.debug
        )

        if not matched_positions:
            raise ValueError("Could not find any matches")
        
        grid_bbox, grid_shape = get_grid_info(matched_positions)
        self.grid = BaseGrid(grid_bbox, grid_shape, self.window_size)
        if not self.grid_path.parent.exists():
            self.grid_path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_saved_grid:
            self.grid.save(str(self.grid_path))
        
        if self.debug:
            print(f'found {len(matched_positions)} matches')
            print(f'Grid shape {self.grid.row} x {self.grid.col}')
            print(f'cell size: {self.grid.symbol_width} x {self.grid.symbol_height}')
            img = draw_grid_on_image(img, self.grid)

        elapsed_time = time.time() - start_time
        print(f"Time taken to initialize grid: {elapsed_time:.2f} seconds")

        if old_grid is not None:
            if old_grid.col == self.grid.col and old_grid.row == self.grid.row:
                self.grid_is_stabilized = True
                print("The grid is stabilized")
            else:
                print("The grid is not stabilized! Try to modify the config")

        if self.debug:
            for pos in matched_positions:
                img = cv2.circle(img, (int(pos[0]), int(pos[1])), 5, (0, 0, 255), -1)
            for template_obj in self.all_templates:
                if template_obj.best_scale is not None:
                    print(f'{template_obj.name} best_scale: {template_obj.best_scale:.3f} score: {template_obj.match_score:.3f}')
                else:
                    print(f'{template_obj.name} best_scale: None')
            print()
            # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            # roi = cv2.resize(roi, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("grid", img)
            cv2.imshow("roi", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # reset best_scale for all templates
        for template_obj in self.all_templates:
            template_obj.best_scale = None
        
    def recognize_roi(self, img):
        if self.grid is None:
            raise ValueError("Grid has not been initialized")
        self.grid.clear()

        for j in range(self.grid.col):
            for i in range(self.grid.row):
                if self.debug:
                    print(f"Processing roi({i}, {j})")
                roi = self.grid.get_roi(i, j)
                # print(self.grid.window_size)
                x, y, w, h = roi
                x1, x2 = x - self.cell_border, x + w + self.cell_border
                y1, y2 = y - self.cell_border, y + h + self.cell_border
                cell_with_border = img[y1:y2, x1:x2]
                cell = img[y:y+h, x:x+w]
                
                if self.cell_matching_method == 0:
                    """ template only """
                    matched_obj, score = process_template_matches(
                        template_list=self.all_templates,
                        roi=cell_with_border,
                        **self.cell_matching_params,
                        grayscale=self.cell_matching_mode=="gray",
                        debug=self.debug
                    )
                
                elif self.cell_matching_method == 1:
                    """ template first """
                    matched_obj, score = process_template_matches(
                        template_list=self.all_templates,
                        roi=cell_with_border,
                        **self.cell_matching_params,
                        grayscale=self.cell_matching_mode=="gray",
                        debug=self.debug
                    )
                    # when template matching fails, use SIFT
                    if matched_obj is None:
                        if self.debug:
                            print(f'template matching failed, using SIFT')
                        matched_obj, score = process_template_matches_sift(
                            template_list=self.all_templates,
                            roi=cell, 
                            **self.sift_matching_params,
                            debug=self.debug
                        )
                        if self.debug and matched_obj is not None:
                            print(f"roi({i}, {j}) symbol: {matched_obj.name}, score: {score}")
                            cv2.imshow("cell", cell)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                
                elif self.cell_matching_method == 2:
                    """ SIFT first """
                    matched_obj, score = process_template_matches_sift(
                        template_list=self.all_templates,
                        roi=cell, 
                        **self.sift_matching_params,
                        debug=self.debug
                    )

                    # when SIFT fails, use template matching
                    if matched_obj is None:
                        if self.debug:
                            print(f'SIFT failed, using template matching')
                        matched_obj, score = process_template_matches(
                            template_list=self.all_templates,
                            roi=cell_with_border,
                            **self.cell_matching_params,
                            grayscale=self.cell_matching_mode=="gray",
                            debug=self.debug
                        )
                        if self.debug and matched_obj is not None:
                            print(f"roi({i}, {j}) symbol: {matched_obj.name}, score: {score}")
                            cv2.imshow("cell", cell)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                if matched_obj is None:
                    if self.debug:
                        cv2.imshow("cell", cell)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    self.grid[i, j] = {"symbol": None, "score": None, "value": None}
                    continue
                self.grid[i, j] = {"symbol": matched_obj.name, "score": score, "value": None}
        
    
    def save_grid_results(self, file_name):
        if self.grid_is_stabilized:
            self.grid.save_results_as_json(save_dir=self.output_json_dir, template_dir=self.template_dir, file_name=file_name)
        
    def save_annotated_frame(self, img, file_name):
        save_path = self.save_dir / f"{file_name}.png"
        draw_bboxes_and_icons_on_image(img, self.template_dir, self.grid, save_path=save_path)

        
class BullGridRecognizer(BaseGridRecognizer):
    def __init__(self, game:str, mode:str, config_file: Path, window_size=(1920, 1080), debug=False):
        super().__init__(game, mode, config_file, window_size, debug)
        self.is_init_column_heights = False
        self.direction = 'up' if self.mode == 'base' else 'down'
        self.arrow_scale_range = self.config["arrow_scale_range"]
        
    def load_templates(self):
        super().load_templates()
        arrow_name = "up_arrow" if self.mode == 'base' else "down_arrow"
        for template_obj in self.all_templates:
            if template_obj.name == arrow_name:
                self.arrow_template_obj = template_obj
                break
        if self.arrow_template_obj is None:
            raise ValueError(f"Arrow template '{arrow_name}' not found in templates.")
        
    def initialize_grid(self, img):
        if self.grid is not None:
            return
        super().initialize_grid(img)
        growth_direction = 'up' if self.mode == 'base' else 'down'
        self.grid = BullGrid(
            bbox=self.grid.bbox, 
            shape=(self.grid.row, self.grid.col),
            growth_direction=growth_direction
        )
            
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
            row_range = (
                range(self.grid.row - self.grid.column_heights[j], self.grid.row) if self.mode == 'base'
                else range(self.grid.column_heights[j])
            )
            
            for i in row_range:
                roi = self.grid.get_roi(i, j)
                x, y, w, h = roi
                x1, x2 = x - self.cell_border, x + w + self.cell_border
                y1, y2 = y - self.cell_border, y + h + self.cell_border
                cell_with_border = img[y1:y2, x1:x2]
                
                matched_obj, score = process_template_matches(
                    template_list=self.all_templates,
                    roi=cell_with_border,
                    **self.cell_matching_params,
                    
                    debug=self.debug
                )
                
                if matched_obj is None:
                    # if self.debug:
                    #     cv2.imshow("cell", cell_with_border)
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()
                    self.grid[i, j] = None
                    continue
                self.grid[i, j] = {"symbol": matched_obj.name, "score": score, "value": None}
        
        if not self.is_init_column_heights:
            self.grid.init_column_heights()
            self.is_init_column_heights = True
            print("initial column heights:", self.grid.column_heights)
        
        #find arrow at each column and update column heights        
        for j in range(self.grid.col):
            if self.mode == 'base':
                index = self.grid.row - self.grid.column_heights[j] - 1
                position = 'top'
            elif self.mode == 'free':
                index = self.grid.column_heights[j]
                position = 'bottom'

            params = self.cell_matching_params.copy()
            params['scale_range'] = self.arrow_scale_range
            roi = self.grid.get_roi(index, j)
            x, y, w, h = roi
            cell = img[y:y+h, x:x+w]
            matched_obj, score = process_template_matches(
                template_list=[self.arrow_template_obj],
                roi=cell,
                **params,
                debug=self.debug
            )
            
            if matched_obj is not None:
                self.grid.column_heights[j] += 1
                if self.grid.column_heights[j] > self.grid.max_height:
                    self.grid.column_heights[j] = self.grid.base_height
                print(f'{self.direction} arrow found at column[{j}], update column_heights[{j}] to {self.grid.column_heights[j]}')

                if self.grid.column_heights[j] > self.grid.row:
                    self.grid.add_row(position=position)
                    print(f'Updated grid shape: {self.grid.row} x {self.grid.col}')
                
        print("column heights:", self.grid.column_heights)