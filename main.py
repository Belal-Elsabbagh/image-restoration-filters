from timeit import default_timer

from src.filters import HarmonicMeanFilter
from src.image_histogram import plot_image_histogram
from src.image_ops import read_img_grayscale, show_img

if __name__ == '__main__':
    img_path = 'data/img/noisy_cameraman.png'
    img = read_img_grayscale(img_path)
    print(img)
    plot_image_histogram(f"{img_path} before", img)
    new_filter = HarmonicMeanFilter(img, 3)
    start = default_timer()
    img = new_filter.get_filtered_img()
    print(f'Time elapsed: {default_timer() - start} seconds')
    plot_image_histogram(f"{img_path} after", img)
    show_img('test', img)
