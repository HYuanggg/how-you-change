import os
import cv2
from tqdm import tqdm
import moviepy.video.io.ImageSequenceClip

dir_I = './data/raw/'
dir_O = './data/edited/'

casc_path = './data/classifier/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(casc_path)

w, h = 140, 140

scale_percent = float(input("输入图片缩放比例（1-100）："))
fps = float(input("输入播放速度（>0，比如5）:"))

print("editing your pictures......")
for direction, _, files in os.walk(dir_I):
    for file in tqdm(files):
        imagePath = direction + file
        image = cv2.imread(imagePath)
        # scale_percent = 30  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # cv2.imshow("pic", image)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
            # minSize=(30, 30)
            # flags=cv2.CV_HAAR_SCALE_IMAGE
        )

        # for (x, y, w, h) in faces:
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("Faces found", image)
        # cv2.waitKey(0)

        x, y, _, _ = faces[0]
        cv2.imwrite(dir_O+file, image[y-h:y+2*h, x-w:x+2*w])

os.system("pause")

# this method raises some error when running as an exe
# out = None
# for direction, _, files in os.walk(dir_O):
#     for file in files:
#         img = cv2.imread(direction+file)
#         # cv2.imshow("Faces found", img)
#         # cv2.waitKey(0)
#         if not out:
#             height, width, _ = img.shape
#             size = (width, height)
#             out = cv2.VideoWriter('./data/video/me.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#         out.write(img)
#
# cv2.destroyAllWindows()
# out.release()

image_files = [dir_O+'/'+img for img in os.listdir(dir_O)]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('./data/video/me.mp4')

os.system("pause")
