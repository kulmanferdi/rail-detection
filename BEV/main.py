import birdseyeview
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./dataset/web/rail-test1.jpg')

bev_image = birdseyeview.birdseye_view(image)

cv2.imshow('birdseye', bev_image)
plt.imshow(bev_image)
plt.savefig('./output/birdseye_view_output.png')
cv2.waitKey(0)
cv2.destroyAllWindows()