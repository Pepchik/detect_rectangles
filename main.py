import cv2
import numpy as np


def main(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    count_squares = 0
    rectangles = []

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            angles = []
            for i in range(4):
                pt1, pt2, pt3 = np.array(approx[i][0]), np.array(approx[(i + 1) % 4][0]), np.array(approx[(i + 2) % 4][0])
                vector1 = pt2 - pt1
                vector2 = pt3 - pt2
                cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  
                angles.append(np.degrees(angle)) 

            if all(80 < angle < 100 for angle in angles):
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                area = width * height

                rectangles.append((approx, area))

    rectangles = sorted(rectangles, key=lambda x: x[1], reverse=True)

    for i, (approx, area) in enumerate(rectangles):
        if i == 0:
            color = (0, 0, 255)  
        elif i == len(rectangles) - 1:
            color = (255, 0, 0) 
        else:
            color = (0, 255, 0)  
        
        cv2.drawContours(img, [approx], 0, color, 2)
        count_squares += 1

    cv2.imshow('Rectangles Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'Count of rectangles: {int(count_squares/2)}')


if __name__ == '__main__':
    main(r'assets\image.png')
