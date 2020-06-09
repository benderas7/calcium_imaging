# Import necessary modules
import cv2
import os
import matplotlib.pyplot as plt

# Set constants
IMAGES_DIR = 'data/N.01.01/images'
VIDEO_FN = 'data/N.01.01/video.tif'


def main(images_dir=IMAGES_DIR, video_fn=VIDEO_FN):
    imgs = [img for img in os.listdir(images_dir) if img.endswith('.tif')]
    frame = plt.imread(os.path.join(images_dir, imgs[0]))[:, :, ::-1]
    print(frame.shape)
    h, w, _ = frame.shape

    video = cv2.VideoWriter(video_fn, 0, 1, (w, h))
    for img in imgs:
        video.write(plt.imread(os.path.join(images_dir, img))[:, :, ::-1])

    cv2.destroyAllWindows()
    video.release()
    return


if __name__ == '__main__':
    main()
