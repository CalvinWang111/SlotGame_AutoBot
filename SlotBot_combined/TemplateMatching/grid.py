class BaseGrid:
    def __init__(self, bbox, shape):
        self.bbox = bbox
        self.row, self.col = shape
        x, y, w, h = bbox
        self.symbol_width = w // self.col
        self.symbol_height = h // self.row
        self._grid = [[None for _ in range(self.col)] for _ in range(self.row)]
        
    def get_roi(self, i, j):
        x, y, w, h = self.bbox
        return (int(x + self.symbol_width * j), int(y + self.symbol_height * i), int(self.symbol_width), int(self.symbol_height))
    
    def clear(self):
        for i in range(self.row):
            for j in range(self.col):
                self._grid[i][j] = None
    
    def __getitem__(self, idx):
        i, j = idx
        return self._grid[i][j]

    def __setitem__(self, idx, value):
        i, j = idx
        self._grid[i][j] = value
    
class BullGrid(BaseGrid):
    def __init__(self, bbox, shape, growth_direction='up'):
        super().__init__(bbox, shape)
        self.growth_direction = growth_direction  # To record the current growth direction
        self.column_heights = [self.row] * self.col  # Track the heights of each column
        self.base_height = 3
        self.max_height = 7
    
    def init_column_heights(self):
        # Initialize the column heights based on the growth direction
        if self.growth_direction == 'up':
            for j in range(self.col):
                for i in range(self.row):
                    if self._grid[i][j] is not None:
                        self.column_heights[j] = self.row - i
                        break
        elif self.growth_direction == 'down':
            for j in range(self.col):
                for i in range(self.row-1, -1, -1):
                    if self._grid[i][j] is not None:
                        self.column_heights[j] = i + 1
                        break

    def add_row(self, position='top'):
        x, y, w, h = self.bbox
        self.row += 1

        # Adjust the bounding box and grid based on the position
        if position == 'top':
            y -= self.symbol_height
            h += self.symbol_height
            self.bbox = (x, y, w, h)
            self._grid.insert(0, [None for _ in range(self.col)])
        elif position == 'bottom':
            h += self.symbol_height
            self.bbox = (x, y, w, h)
            self._grid.append([None for _ in range(self.col)])
        else:
            raise ValueError("position must be 'top' or 'bottom'")