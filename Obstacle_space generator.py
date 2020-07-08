# =========================================================================================================================================================================================================================== #
# -------------> Project 03 <---------------#
# =========================================================================================================================================================================================================================== #
# Authors   :-> Sudharsan
# Date      :-> 23 April 2019
# =========================================================================================================================================================================================================================== #

# =========================================================================================================================================================================================================================== #
# Import Section
# =========================================================================================================================================================================================================================== #
import numpy as np
import cv2 as cv
import os, sys, time, math


# =========================================================================================================================================================================================================================== #
# Obstacle Space Class Definition
# =========================================================================================================================================================================================================================== #
class ObstacleSpace:
    def __init__(self, width: int, height: int, res=1.0) -> None:
        self.res = res
        self.obstacle_space = np.ones((height, width, 3))
        self.robot_points = []
        self.obstacles = set()
        self.coordinates = set()
        self.circle_minowski = set()
        self.net_obstacle = set()
        self.minowski = set()
        clearance = int(res * 22)

        t0 = time.time()

        # -----> Generate Robot Space <----- #
        print('Robot Point Creation: started..!')
        for y in range(-clearance, clearance + 1):
            for x in range(-clearance, clearance + 1):
                if x ** 2 + y ** 2 - clearance ** 2 <= 0:
                    self.robot_points.append((y, x))

        print('Robot Points Creation: Done..!')

        # -----> Generate Obstacle Space <----- #
        print('Obstacle Space Creation: started..!')
        for y in range(height):
            for x in range(width):
                self.rect_check(y, x)
                self.circle_check(y, x, 0)
                self.boundry_check(y, x)
                self.circle_check(y, x, 29)
                if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                    self.obstacle_space[y, x] = (0.6, 0.6, 0)
                    if (y, x) not in self.obstacles:
                        self.obstacles.add((y, x))
                    if (y, x) not in self.coordinates:
                        self.coordinates.add((y, x))

        t1 = time.time()

        print('Obstacle Creation: Done..!')
        print("TIme Elapsed: ", t1 - t0)

        # -----> Generate Minowski summed Obstacle Space <----- #
        print('Minowski sum Obstacle Creation: started..!')
        self.minowski_sum(self.robot_points, height, width)

        t2 = time.time()
        print('Minowski sum obstacle Creation: Done..!')
        print("Total TIme Elapsed: ", t2 - t0)

    # ======================================================================================================================================================================================================================= #
    # Function for rectange check
    # ======================================================================================================================================================================================================================= #
    def rect_check(self, y: int, x: int):
        if 150 <= x <= 309.7 and 100 <= y <= 260:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 01
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 832 <= x <= 918 and 0 <= y <= 183:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 02
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 983 <= x <= 1026 and 0 <= y <= 91:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 03
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 744 <= x <= 1111 and 313 <= y <= 389:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 04
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 1052 <= x <= 1111 and 444.5 <= y <= 561.5:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 05
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 1019 <= x <= 1111 and 561.5 <= y <= 647.5:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 06
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 784.5 <= x <= 934.5 and 626 <= y <= 743:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 07
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 529 <= x <= 712 and 669 <= y <= 745:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 08
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 438 <= x <= 529 and 512 <= y <= 695:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 09
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 1052 <= x <= 1111 and 714.8 <= y <= 831.8:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 10
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 927 <= x <= 1111 and 899 <= y <= 975:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 11
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 779 <= x <= 896 and 917 <= y <= 975:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 12
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 474 <= x <= 748 and 823 <= y <= 975:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 13
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))
        elif 685 <= x <= 1111 and 975 <= y <= 1011:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 14
            if (y, x) not in self.obstacles: self.obstacles.add((y, x))

    # ======================================================================================================================================================================================================================= #
    # Function for Circle check
    # ======================================================================================================================================================================================================================= #
    def circle_check(self, y: int, x: int, flag=0):
        if flag:
            if (39.5) ** 2 < ((x - 390) ** 2 + (y - 45) ** 2) < (41.5) ** 2:
                self.coordinates.add((y, x))
            elif (39.5) ** 2 < ((x - 438) ** 2 + (y - 274) ** 2) < (41.5) ** 2:
                self.coordinates.add((y, x))
            elif (39.5) ** 2 < ((x - 438) ** 2 + (y - 736) ** 2) < (41.5) ** 2:
                self.coordinates.add((y, x))
            elif (39.5) ** 2 < ((x -
                                 390) ** 2 + (y - 965) ** 2) < (41.5) ** 2:
                self.coordinates.add((y, x))
            elif (79) ** 2 < ((x - 150) ** 2 + (y - 180) ** 2) < (81) ** 2:
                self.coordinates.add((y, x))
            elif (79) ** 2 < ((x - 309.7) ** 2 + (y - 180) ** 2) < (81) ** 2:
                self.coordinates.add((y, x))
        else:
            if ((x - 390) ** 2 + (y - 45) ** 2) <= (41) ** 2:
                self.obstacles.add((y, x))
                self.obstacle_space[y, x] = [0.6, 0.6, 0]

            elif ((x - 438) ** 2 + (y - 274) ** 2) <= (41) ** 2:
                self.obstacles.add((y, x))
                self.obstacle_space[y, x] = [0.6, 0.6, 0]

            elif ((x - 438) ** 2 + (y - 736) ** 2) <= (41) ** 2:
                self.obstacles.add((y, x))
                self.obstacle_space[y, x] = [0.6, 0.6, 0]

            elif ((x - 390) ** 2 + (y - 965) ** 2) <= (41) ** 2:
                self.obstacles.add((y, x))
                self.obstacle_space[y, x] = [0.6, 0.6, 0]

            elif ((x - 150) ** 2 + (y - 180) ** 2) <= (80) ** 2:
                self.obstacles.add((y, x))
                self.obstacle_space[y, x] = [0.6, 0.6, 0]

            elif ((x - 309.7) ** 2 + (y - 180) ** 2) <= (80) ** 2:
                self.obstacles.add((y, x))
                self.obstacle_space[y, x] = [0.6, 0.6, 0]

    # ======================================================================================================================================================================================================================= #
    # Function for Boundry check to perform minowski
    # ======================================================================================================================================================================================================================= #
    def boundry_check(self, y: int, x: int):
        if ((150 == x or x == 310) and 100 <= y <= 260) or (
                150 <= x <= 310 and (100 == y or y == 260)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 01
        if ((832 == x or x == 918) and 0 <= y <= 183) or (
                832 <= x <= 918 and (0 == y or y == 183)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 02
        if ((983 == x or x == 1026) and 0 <= y <= 91) or (
                983 <= x <= 1026 and (0 == y or y == 91)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 03
        if ((744 == x or x == 1111) and 313 <= y <= 389) or (
                744 <= x <= 1111 and (313 == y or y == 389)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 04
        if ((1052 == x or x == 1111) and 445 <= y <= 562) or (
                1052 <= x <= 1111 and (445 == y or y == 562)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 05
        if ((1019 == x or x == 1111) and 562 <= y <= 648) or (
                1019 <= x <= 1111 and (562 == y or y == 648)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 06
        if ((785 == x or x == 935) and 626 <= y <= 743) or (
                785 <= x <= 935 and (626 == y or y == 743)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 07
        if ((529 == x or x == 712) and 669 <= y <= 745) or (
                529 <= x <= 712 and (669 == y or y == 745)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 08
        if ((x == 438 or x == 529) and 512 <= y <= 695) or (
                438 <= x <= 529 and (y == 512 or y == 695)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 09
        if ((x == 1052 or x == 1111) and 715 <= y <= 832) or (
                1052 <= x <= 1111 and (y == 715 or y == 832)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 10
        if ((x == 927 or x == 1111) and 899 <= y <= 975) or (
                927 <= x <= 1111 and (y == 899 or y == 975)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 11
        if ((x == 779 or x == 896) and 917 <= y <= 975) or (
                779 <= x <= 896 and (y == 917 or y == 975)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 12
        if ((x == 474 or x == 748) and 823 <= y <= 975) or (
                474 <= x <= 748 and (y == 823 or y == 975)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 13
        if ((x == 685 or x == 1111) and 975 <= y <= 1011) or (
                685 <= x <= 1111 and (y == 975 or y == 1011)): self.coordinates.add(
            (y, x))  # Boundry check for Rectangle: 14

    # ======================================================================================================================================================================================================================= #
    # Function for Minowski Sum 
    # ======================================================================================================================================================================================================================= #
    def minowski_sum(self, robot_points: np.array, h: int, w: int) -> None:
        for y, x in self.coordinates:
            for j in robot_points:
                point = (y - (-1) * j[0], x - (-1) * j[1])
                if 0 <= point[1] < w and 0 <= point[0] < h:
                    if point not in self.obstacles:
                        self.minowski.add(point)
                        self.obstacle_space[point] = [0.6, 0.6, 0.6]

        self.net_obstacle = self.obstacles.union(self.minowski)
        print(len(self.obstacles))
        print(len(self.minowski))
        print(len(self.net_obstacle))


if __name__ == '__main__':
    res = 1
    maps = ObstacleSpace(int(1111 * res), int(1011 * res), res)
    cv.imshow("Maps", maps.obstacle_space)
    maps.obstacle_space = 255 * maps.obstacle_space
    obstacle_space = maps.obstacle_space.astype('uint8')
    cv.imwrite('./Data/Obstacle_space.jpg', obstacle_space)
    obstacle_space_gray = cv.cvtColor(obstacle_space, cv.COLOR_BGR2GRAY)
    cv.imwrite('./Data/Obstacle_space_gray.jpg', obstacle_space_gray)
    obstacle_space_binary = (255 * (obstacle_space_gray == 255)).astype('uint8')
    cv.imwrite('./Data/Obstacle_space_binary.jpg', obstacle_space_binary)
    cv.waitKey(0)
