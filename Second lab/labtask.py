import cv2 as cv
import matplotlib.pyplot as plt

img1=cv.imread("image1.jpg")
gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

img2=cv.imread("image2.jpg")
gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)


img3=cv.imread("image3.jpg")
gray3=cv.cvtColor(img3,cv.COLOR_BGR2GRAY)


img4=cv.imread("image4.jpg")
gray4=cv.cvtColor(img4,cv.COLOR_BGR2GRAY)


img5=cv.imread("image5.jpg")
gray5=cv.cvtColor(img5,cv.COLOR_BGR2GRAY)

cv.imshow("image3",gray5)
cv.waitKey(0)


hist1 = cv.calcHist([gray1], [0], None, [256], [0, 256])
hist2 = cv.calcHist([gray2], [0], None, [256], [0, 256])
hist3 = cv.calcHist([gray3], [0], None, [256], [0, 256])
hist4 = cv.calcHist([gray4], [0], None, [256], [0, 256])
hist5 = cv.calcHist([gray5], [0], None, [256], [0, 256])




# plt.subplot(121)
# plt.title("Image1")
# plt.xlabel('bins')
# plt.ylabel("No of pixels")
# plt.plot(hist1)
# plt.show()



# plt.subplot(121)
# plt.title("Image2")
# plt.xlabel('bins')
# plt.ylabel("No of pixels")
# plt.plot(hist2)
# plt.show()



# plt.subplot(121)
# plt.title("Image3")
# plt.xlabel('bins')
# plt.ylabel("No of pixels")
# plt.plot(hist3)
# plt.show()



# plt.subplot(121)
# plt.title("Image4")
# plt.xlabel('bins')
# plt.ylabel("No of pixels")
# plt.plot(hist4)
# plt.show()



# plt.subplot(121)
# plt.title("Image5")
# plt.xlabel('bins')
# plt.ylabel("No of pixels")
# plt.plot(hist5)
# plt.show()

