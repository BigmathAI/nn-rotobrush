from data_layer import data_layer_2d
from config import FLAGS, logger
import os, cv2, time

def main():
    dl_train = data_layer_2d(FLAGS, 'train')
    dl_valid = data_layer_2d(FLAGS, 'valid')

if __name__ == '__main__':
    main()