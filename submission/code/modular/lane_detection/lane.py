import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import os

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)

    match_mask_color = (255,) * 3
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2, itype='gray'):
    # If there are no lines to draw, exit.
    if lines is None:
        return img
    
    # Make a copy of the original image.
    img = np.copy(img) 

    # Create a blank image that matches the original in size.
    if itype == 'gray':
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
            ),
            dtype=np.uint8,
        )    
    else:
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
                3
            ),
            dtype=np.uint8,
        )    

    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)    
            
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)    

    # Return the modified image.
    return img

def find_lines(cannyed_image):
    lines = cv2.HoughLinesP(
        cannyed_image,
        rho=6,
        theta=np.pi / 200,
        threshold=160,
        lines=np.array([]),
        minLineLength=20,
        maxLineGap=25
    )

    return lines

def is_sky(rgb_img, x0, y0):
    return rgb_img[y0][x0][0] == 92 \
        and rgb_img[y0][x0][1] == 167 \
        and rgb_img[y0][x0][2] == 204

def is_ignore(rgb_img, x, y):
    road_const = 75

    return is_sky(rgb_img, x, y) \
        or rgb_img[y][x][0] > 100 \
        or rgb_img[y][x][1] > 90 \
        or (rgb_img[y][x][0] < road_const \
            and rgb_img[y][x][1] < road_const \
            and rgb_img[y][x][2] < road_const)

def is_red(rgb_img, x, y):
    red_const1 = 100
    red_const2 = 50

    return rgb_img[y][x][0] >= red_const1 \
        and rgb_img[y][x][1] < red_const2 \
        and rgb_img[y][x][2] < red_const2

def is_red_bar(rgb_img, x, y, xx, yy):
    return is_red(rgb_img, x, y) or is_red(rgb_img, xx, yy)

def is_line_on_road(rgb_img, x0, y0, x1, y1):
    return is_red_bar(rgb_img, x0, y0, x1, y1) \
        or (is_ignore(rgb_img, x0, y0) \
        and is_ignore(rgb_img, x1, y1))

def is_yellow(r, g, b):
    return r >= 150 and g >= 150 and b < 150 and r > g and (g - b) >= 50

def is_white_grayish(r, g, b):
    return r >= 120 and g >= 120 and b >= 120 \
        and abs(np.int32(r) - np.int32(g)) < 10 \
        and abs(np.int32(g) - np.int32(b)) < 10 \
        and abs(np.int32(b) - np.int32(r)) < 10

def find_red(rgb_img, start_x, start_y, end_x, end_y):
    red_cells = 0
    total_pixels = (end_y - start_y + 1) * (end_x - start_x + 1)
    y, x = start_y, start_x

    while y <= end_y:
        while x <= end_x:
            r, g, b = rgb_img[y][x][0], rgb_img[y][x][1], rgb_img[y][x][2]
            if r >= 100 and g < 50 and b < 50:
                red_cells += 1

            x += 1
        y += 1
        x = start_x

    return red_cells / total_pixels

def has_red_bar(rgb_img):
    x = find_red(rgb_img, 100, 500, 699, 599)
    # print(x)
    if x > 0.04:
        return True, 'bot'
    
    x = find_red(rgb_img, 100, 400, 699, 599)
    # print(x)
    if x > 0.04:
        return True, 'mid'

    x = find_red(rgb_img, 100, 300, 699, 399)
    # print(x)
    if x > 0.04:
        return True, 'top'

    return False, None

def inspect_box(rgb_img, start_x, start_y, end_x, end_y):
    # assume shape (600, 800, 3)
    yellow_lane_cells = 0
    white_lane_cells = 0
    max_g = 0

    # y, x = 300, 0
    # end_y, end_x = 599, 399
    total_pixels = (end_y - start_y + 1) * (end_x - start_x + 1)
    y, x = start_y, start_x

    while y <= end_y:
        while x <= end_x:
            r, g, b = rgb_img[y][x][0], rgb_img[y][x][1], rgb_img[y][x][2]
            max_g = max(max_g, g)
            if is_yellow(r, g, b):
                yellow_lane_cells += 1
            
            if is_white_grayish(r, g, b):
                white_lane_cells += 1
            
            x += 1

        y += 1
        x = start_x

    return yellow_lane_cells / total_pixels, white_lane_cells / total_pixels, max_g

def is_pit(rgb_img, start_x, start_y, end_x, end_y):
    pit_black = 0

    # y, x = 300, 0
    # end_y, end_x = 599, 399
    total_pixels = (end_y - start_y + 1) * (end_x - start_x + 1)
    y, x = start_y, start_x

    while y <= end_y:
        while x <= end_x:
            r, g, b = rgb_img[y][x][0], rgb_img[y][x][1], rgb_img[y][x][2]
            if r < 30 and g < 30 and b < 30:
                pit_black += 1
            
            x += 1

        y += 1
        x = start_x

    return pit_black / total_pixels

def detect_lane(rgb_array=None, filename=None, show='rgb'):
    # reading in an image
    if rgb_array is None and filename is None:
        return
    
    if filename:
        image = np.load(filename)
    else:
        image = rgb_array

    # print(is_pit(rgb_array, 100, 200, 699, 399))

    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    # exit()

    # image = image.transpose(1, 0, 2)

    # printing out some stats and plotting the image

    # print('This image is:', type(image), 'with dimensions:', image.shape)

    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertice_sets = [
        [
            [0, 599],
            [0, 400],
            [400, 100],
            [799, 400],
            [799, 599]
        ],

        [
            [0, 599],
            [0, 300],
            [400, 200],
            [799, 300],
            [799, 599]  
        ],

        [
            [0, height-1],
            [0, int(2 * height/3)],
            [int(2 * height/3), 100],
            [width-1, int(2 * height/3)],
            [width-1, height-1]
        ]
    ]

    # Convert to grayscale here.
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Call Canny Edge Detection here.
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertice_sets[0]], np.int32)
    )

    lines_raw = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 200,
        threshold=150,
        lines=np.array([]),
        minLineLength=10,
        maxLineGap=25
    )

    # print('lines_raw: ', lines_raw)
    if lines_raw is None:
        return [], None, None

    lines = []
    pos_slope = []
    neg_slope = []
    for line in lines_raw:
        skip = False
        for x0, y0, x1, y1 in line:
            if (x0 == x1) or (y0 == y1) or (y0 < 195 and y1 < 195) or is_line_on_road(image, x0, y0, x1, y1):
                skip = True
            else:
                # print((x0, y0, x1, y1), image[y0][x0], image[y1][x1])
        
                slope = (y0 - y1)/(x1 - x0)
                skip = abs(slope) < 0.05
        
        if not skip:
            lines.append(line)
            if slope > 0:
                pos_slope.append(slope)
            else:
                neg_slope.append(slope)
            
            # print("--Slope", slope)

    pos_slope_avg, neg_slope_avg = None, None

    if len(pos_slope) > 0:
        pos_slope_avg = sum(pos_slope) / len(pos_slope)

    if len(neg_slope) > 0:
        neg_slope_avg = sum(neg_slope) / len(neg_slope)

    # print("---Avg slopes: ", pos_slope_avg, neg_slope_avg)
    # print(inspect_box(image, 0, 300, 399, 599))
    # print(inspect_box(image, 400, 300, 799, 599))

    lines = np.array(lines)

    # print(has_red_bar(image))

    # line_image_cannyed = draw_lines(cannyed_image, lines, thickness=5, itype='gray')

    # line_image_rgb = draw_lines(image, lines, thickness=5, itype='rgb')
    
    # to_show = line_image_rgb if show == 'rgb' else line_image_cannyed

    # plt.figure()

    # plt.imshow(to_show)
    # plt.show()

    return lines, pos_slope_avg, neg_slope_avg


if __name__ == '__main__':

    cnt = 0
    for f in os.listdir("true_camera/middle"):
        if '.npy' not in f:
            continue
         
        if cnt < 0:
            cnt += 1
            continue

        print(f)
        filename = os.path.join('true_camera/middle', f)
        detect_lane(filename=filename, show='rgb')
        cnt += 1
        if cnt >= 14:
            break
