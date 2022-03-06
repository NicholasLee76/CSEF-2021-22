import matplotlib.pyplot as plt
import numpy as np
import cv2

def main(run, clear):
    coords_file = open("./" + run + "/coords.txt", "r")
    full_text = coords_file.read()
    coords = full_text.split("\n")
    coords.pop()
    x = [0] * len(coords)
    y = [0] * len(coords)
    for i in range(1, len(coords)):
        nums = coords[i].split(',')
        x[i] = int(nums[1]) * 0.05
        y[i] = -int(nums[2]) * 0.05

    fig, axs = plt.subplots(1,1)
    axs.set_aspect('equal', 'box')
    min_tick = min(int(min(x)-1), int(min(y)-1))
    max_tick = max(int(max(x)+1), int(max(y)+1))


    pieces = 15
    split_length = int(len(coords)/pieces)
    i = 0
    while i < len(coords):
        if i+split_length <= len(coords):
            split = [x for x in range(i, i+split_length)]

        else:
            split = [x for x in range(i, len(coords))]

        # print(split)
        split_x = [x[j] for j in split]
        split_y = [y[j] for j in split]
        # print(split_x)
        plt.plot(split_x, split_y)
        if clear:
            plt.savefig("./" + run + "/" + run + "_split" + str(int(split[0]/split_length)) + ".png")
            plt.clf()
        else:
            if int(split[0]/split_length) == pieces:
                plt.xticks(range(-29, 7, 5))
                plt.yticks(range(-29, 7, 5))
                plt.savefig("./" + run + "/full_splits.png")
        i += split_length

    if clear:
        for k in range(pieces+1):
            img = cv2.imread("./" + run + "/" + run + "_split" + str(k) + ".png", cv2.IMREAD_GRAYSCALE)
            thresh, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            crop = binary_img[60:424, 80:560]
            square = cv2.resize(crop, (364, 364))
            cv2.imwrite("./" + run + "/" + run + "_split" + str(k) + ".png", square)


if __name__ == '__main__':
    main('olf_test1', False)
