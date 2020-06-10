# Import necessary modules
import caiman_demo

# Set constants
DATA_FILE = 'data/NeuroPAL/11.25.19' \
            '/worm3_gcamp_Out/worm3_gcamp_Out_t001z01_ORG.tif'


def main(file=DATA_FILE):
    caiman_demo.main(video_fn=file)
    return


if __name__ == '__main__':
    main()
