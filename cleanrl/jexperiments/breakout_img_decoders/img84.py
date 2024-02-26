from dataclasses import dataclass
import tyro
import matplotlib.pyplot as plt

photos_saved = 0

@dataclass
class ImageArgs:
    #Image
    image_size: int = 128
    """width and height of the image"""
    
    # Bricks
    wall_size: int = 5
    """the size of the wall in the image"""
    brick_columns: tuple = (2,6,10,14,18,22,27,31,35,39,43,47,51,55,59,63,67,72)
    """where the bricks are located in the image, starting from first pixel after the wall"""
    brick_values : tuple = (85, 124, 148, 131, 129, 110)
    """brick pixel values, bottom up"""
    brick_lines : tuple = (36, 33, 31, 29, 26, 23)
    """horizontal lines where bricks info is located"""
    
    #Bar
    bar_line: int = 76
    """horizontal line where the bar is located"""
    bar_value : int = 110
    """bar pixel value on line 'bar_line'"""
    bar_half_size: int = 5
    """half the size of the bar"""
    
    #Ball
    circular_for_radius: int = 5
    """radius of the circular for loop to find the ball"""
    ball_line_start: int = 13
    """first line where the ball can be found"""
    ball_line_end: int = 77
    """last line where the ball can be found"""
    
img_args = tyro.cli(ImageArgs)


def circular_for(start_x, start_y, max_x, max_y, min_x, min_y,radius = 4, fn = lambda x,y: False):
    """
    After benchmarking, this is 3x slower than normal for loop, if iterated on the same array
    If, with this function, I am able to iterate over an array 1/3 of the size, it will be worth it
    """
    x,y = start_x, start_y
    for n in range(1,radius+1):
        y = max(min_y, start_y - n)
        x = start_x
        if fn(x,y):
            return x,y
        right, down, left, up, back = n, n*2, n*(2), n*(2), n-1
        for r in range(right):
            x = min(max_x, x + 1)
            if fn(x,y):
                return x,y
        for d in range(down):
            y = min(max_y, y + 1)
            if fn(x,y):
                return x,y
        for l in range(left):
            x = max(min_x, x - 1)
            if fn(x,y):
                return x,y
        for u in range(up):
            y = max(min_y, y - 1)
            if fn(x,y):
                return x,y
        for b in range(back):
            x = min(max_x, x + 1)
            if fn(x,y):
                return x,y
    return -1,-1
        
def get_bricks_column_value(frame, col_number):
    out_arr = [0,0,0,0,0,0]
    for i, line in enumerate(img_args.brick_lines):
        out_arr[i] = (frame[line][(img_args.wall_size + img_args.brick_columns[col_number])] ==\
            img_args.brick_values[i])
    return sum(map(lambda x: x[1]*2**x[0], enumerate(out_arr)))

def get_bar_pos(frame):
    line = frame[img_args.bar_line][img_args.wall_size:(img_args.image_size-img_args.wall_size)]
    start_pixel = 0
    end_pixel = 0
    found = False
    first_empty = line[0] != img_args.bar_value
    for i, pixel in enumerate(line):
        if not(found):
            if pixel == img_args.bar_value:
                start_pixel = i
                found = True
        else:
            if pixel != img_args.bar_value:
                end_pixel = i
                break
    if first_empty:
        return start_pixel + img_args.bar_half_size + img_args.wall_size
    else:
        return end_pixel - img_args.bar_half_size + img_args.wall_size
    
def get_ball_pos(frame):
    global last_ball_pos
    
    def is_ball(column,line):
        lst = [frame[line-1][column], frame[line][column+1], frame[line+1][column], frame[line][column-1]]
        val = frame[line][column]
        if len(list(filter(lambda x: 0< x < val, lst))) >= 2:
            lst = [frame[line-2][column], frame[line][column+2], frame[line+2][column], frame[line][column-2]]
            return len(list(filter(lambda x: x == 0, lst))) >= 2
        return False
    
    if last_ball_pos[0] < 0:
        # Ball was not found in the frame before, must do a for loop in the entire ball space
        for line in range(img_args.ball_line_start, img_args.ball_line_end +1 ):
            for column in range(img_args.wall_size, img_args.image_size - img_args.wall_size):
                if is_ball(column, line,):
                    last_ball_pos = (column, line)
                    return column, line
        last_ball_pos = (-1,-1)
        return -1,-1
    else:
        # Ball was found in the frame before, will do a circular for loop around the last position
        last_ball_pos = circular_for(last_ball_pos[0], last_ball_pos[1], img_args.image_size-img_args.wall_size-1, 
                                     img_args.ball_line_end,img_args.wall_size, img_args.ball_line_start,
                                     radius = img_args.circular_for_radius,fn = is_ball)   
        return last_ball_pos
    
def get_ball_speed(prev_frame,next_frame):
    # last_ball_pos has the last position of the ball, that is calculated before calling this function
    global last_ball_pos, photos_saved
    last_ball_pos_b = last_ball_pos
    pos = get_ball_pos(prev_frame)
    if pos[0]<0 or last_ball_pos_b[0]<0:
        if pos[0]<0 and last_ball_pos_b[0]<0:
            return (0,0)
        else:
            return pos if pos[0]>0 else last_ball_pos_b
    return (pos[0] - last_ball_pos_b[0], pos[1] - last_ball_pos_b[1])
        
    
# End of Johnny State