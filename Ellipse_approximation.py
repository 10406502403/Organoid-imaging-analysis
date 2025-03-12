import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
  
def fit_ellipse_to_mask(mask):  
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    if not contours:  
        raise ValueError("No contours found in mask")  
    max_contour = max(contours, key=cv2.contourArea)  
    if len(max_contour) < 5:  
        raise ValueError("Contour has too few points to fit an ellipse")  
    ellipse = cv2.fitEllipse(max_contour)  
      
    return ellipse  
  
def main():  
    mask = cv2.imread('/home/featurize/work/xhh/dm/image6.jpg', cv2.IMREAD_GRAYSCALE)  
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  
    try:  
        ellipse = fit_ellipse_to_mask(mask)  
    except ValueError as e:  
        print(e)  
        return  

    result_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  
      
    cv2.ellipse(result_image, ellipse, (0, 0, 255),1)  
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)   
    plt.figure(figsize=(10, 5))  
    plt.title('Mask with Ellipse Contour')  
    plt.imshow(result_image)  
    plt.show()  
  
if __name__ == "__main__":  
    main()
