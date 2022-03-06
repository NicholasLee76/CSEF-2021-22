import math

def main(run):
    coords_file = open("./" + run + "/coords.txt", "r")
    full_text = coords_file.read()
    coords = full_text.split("\n")
    coords.pop()
    x = [0] * len(coords)
    y = [0] * len(coords)

    total_distance = 0
    for i in range(1, len(coords)):
        nums = coords[i].split(',')
        x[i] = int(nums[1])
        y[i] = -int(nums[2])
        dx = abs(x[i] - x[i-1])
        dy = abs(y[i] - y[i-1])
        total_distance += math.sqrt(dx ** 2 + dy ** 2)


    total_distance *= (26.5/525.4)
    speed = total_distance / (500/30)
    print(run + ' ' + str(speed))
if __name__ == '__main__':
    main('speed_control_test2')
