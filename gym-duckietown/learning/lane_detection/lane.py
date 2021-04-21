import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    print(type(mask), mask.shape, mask.dtype)

    match_mask_color = (255,) * 3
      
    # Fill inside the polygon
    cv2.fillPoly(np.float32(mask), vertices, match_mask_color)
    
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

def canny_lane(img):
    # Assume float img
    img = np.uint8(255 * img)
    img = img.transpose(1, 2, 0)

    # Convert to grayscale here.
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Call Canny Edge Detection here.
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    cannyed_image = cannyed_image.transpose(2, 0, 1)
    cannyed_image = np.float32(cannyed_image / 255)

    return cannyed_image

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

def detect_lane(filename):
    # reading in an image

    image = np.uint8(255 * np.load(filename))
    image = image.transpose(1, 2, 0)

    # printing out some stats and plotting the image

    print('This image is:', type(image), 'with dimensions:', image.shape)
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (0, 0),
        (0, 40),
        (160, 40),
        (120, 0)
    ]

    # Convert to grayscale here.
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Call Canny Edge Detection here.
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    lines = cv2.HoughLinesP(
        cannyed_image,
        rho=6,
        theta=np.pi / 270,
        threshold=160,
        lines=np.array([]),
        minLineLength=20,
        maxLineGap=25
    )

    print(lines)

    line_image = draw_lines(image, lines, itype='rgb')

    to_show = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )

    plt.figure()

    # plt.imshow(cannyed_image)
    # plt.show()

    plt.imshow(line_image)
    plt.show()


if __name__ == '__main__':
    i, max_iter = 1, 11

    while i <= max_iter:
        filename = 'env{}.npz.npy'.format(i)
        detect_lane(filename)
        i += 1

