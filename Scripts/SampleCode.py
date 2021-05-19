import numpy as np
import cv2

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    ignore_mask_color = (255)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
  img = np.copy(img)
  blank_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

  for line in lines:
    for x1, y1, x2, y2 in line:
      cv2.line(blank_img, (x1,y1), (x2,y2), (0,255, 0), thickness=3)
  
  img = cv2.addWeighted(img, 0.8, blank_img, 1, 0.0)
  return img

def process(image):
  def grayscale(image):
      return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  gray = grayscale(image)
  canny_image = cv2.Canny(gray, 100, 200)
  
  #Applies a Gaussian Noise kernel
  def gaussian_blur(img, kernel_size):
      return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
  gaus_image = gaussian_blur(canny_image, 1)

  height = image.shape[0]
  width = image.shape[1]
  vertices = [(0,height), (width/2, 310), (width, height)]
  masked = region_of_interest(gaus_image, np.array([vertices], np.int32))

  lines = cv2.HoughLinesP(masked,
                          rho=6,
                          theta=np.pi/60,
                          threshold=160,
                          lines=np.array([]),
                          minLineLength=40,
                          maxLineGap=25)
  image_with_lines = draw_lines(image, lines)
  return image_with_lines

cap = cv2.VideoCapture('Udacity_git/test_videos/challenge.MP4')
while(cap.isOpened()):
  ret, frame = cap.read()
  frameWithLines = process(frame)
  cv2.imshow('frameWindow', frameWithLines)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()