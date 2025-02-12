import os
import cv2
import matplotlib.pyplot as plt  


df = 'bt dataset/'
cols=['yes','no']
data1 = os.path.join(df,cols[0])
print(data1)
data2 = os.path.join(df,cols[1])
print(data2)

dt =os.listdir(data1)
print(dt)
dt1 =os.listdir(data2)
print(dt1)

#function 

def loading(img):
    image = cv2.imread(img)
    image=cv2.resize(image(69,69))
    return image[...,::-1] 

plt.figure(figsize=(12, 9))

for i in range(min(12, len(dt1))):  # Ensuring you don't exceed available images
    img_path = os.path.join(data1, dt1[i])
    plt.subplot(3, 4, i + 1)
    plt.imshow(loading(img_path))
    plt.title("yes")
    plt.axis('off')  # Hide the axes

plt.suptitle("Images from 'yes' folder", fontsize=16)
plt.tight_layout()
plt.show()                                      

