from abc import abstractmethod
from functools import wraps

import numpy as np

from src.image_ops import get_shape
from src.image_ops.quantize_intensity import quantize_intensity


def memoize(func):
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


class BaseFilter:
    window_len = 3

    def __init__(self, _img, window_len):
        if window_len % 2 == 0 or window_len < 3:
            raise Exception(f"Window length must be an odd number not less than 3. Given {window_len}")
        self.img = _img
        self.img_shape = get_shape(_img)
        self.window_len = window_len
        self.offset = int(self.window_len / 2)
        self.row_mirror, self.col_mirror = self.init_mirror_indexes(_img, self.offset, self.window_len)

    def set_img(self, _img):
        self.img = _img

    @abstractmethod
    def convulse(self, pixels):
        pass

    @staticmethod
    def init_mirror_indexes(_img, _offset, window_len):
        img_shape = get_shape(_img)
        bounds = BaseFilter.get_index_bounds(img_shape.height, img_shape.width, _offset)
        return BaseFilter.get_mirrors(window_len, img_shape, *bounds)

    def get_pixel(self, _row, _col):
        return self.img[self.row_mirror.get(_row, _row)][self.col_mirror.get(_col, _col)]

    def get_window(self, _row, _col):
        min_row, min_col, max_row, max_col = self.get_window_limits(_row, _col)
        return [[self.get_pixel(r, p) for p in range(min_col, max_col + 1)] for r in range(min_row, max_row + 1)]

    def get_window_limits(self, _row, _col):
        """
        Returns:
             min_row, min_col, max_row, max_col
        """
        return _row - self.offset, _col - self.offset, _row + self.offset, _col + self.offset

    @staticmethod
    def get_index_bounds(_height, _width, _offset):
        init_index = 0 - _offset
        return init_index, init_index, _height + _offset, _width + _offset

    def get_filtered_img(self):
        new_img = [[self.convulse_pix(r, p) for p in range(self.img_shape.width)] for r in range(self.img_shape.height)]
        return np.array(new_img, dtype=np.uint8)

    def convulse_pix(self, row_ind, pix_ind):
        return quantize_intensity(self.convulse(self.get_window(row_ind, pix_ind)))

    @staticmethod
    def get_mirrors(window_len, img_shape, _init_row, _init_col, _end_row, _end_col):
        mirror_offset = window_len - 1
        row_mirror_index = {i: i + mirror_offset for i in range(_init_row, 0)}
        col_mirror_index = {i: i + mirror_offset for i in range(_init_col, 0)}
        row_mirror_index.update({i: i - mirror_offset for i in range(img_shape.height, _end_row)})
        col_mirror_index.update({i: i - mirror_offset for i in range(img_shape.width, _end_col)})
        return row_mirror_index, col_mirror_index
