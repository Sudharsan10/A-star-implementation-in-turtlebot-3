#!/usr/bin/env python
# =========================================================================================================================================================================================================================== #
# -------------> Project 03 <---------------#
# =========================================================================================================================================================================================================================== #
# Authors   :-> Sudharsan(UID: 116298636)
# Course    :-> ENPM661 - Planning for Autonomous Robots
# Date      :-> 23 April 2019
# =========================================================================================================================================================================================================================== #

# =========================================================================================================================================================================================================================== #
# Import Section
# =========================================================================================================================================================================================================================== #
import numpy as np
import cv2 as cv
import os
import sys
import time
import math
import rospy
from geometry_msgs.msg import Twist


# =========================================================================================================================================================================================================================== #

# =========================================================================================================================================================================================================================== #
# Node Class Definition
# =========================================================================================================================================================================================================================== #
class Nodes:
    def __init__(self, current_index, parent_index, goal_index, cost=float('inf')):
        self.theta = 0
        self.cost = cost
        self.h_cost = float('inf')
        self.heuristic_distance = ((current_index[0] - goal_index[0]) ** 2 + (current_index[1] - goal_index[1]) ** 2) ** .5
        self.obstacle = False
        self.index = current_index
        self.parent = parent_index
        self.velocity = (0, 0)
        self.omega = 0
        self.linear_velocity = 0


# =========================================================================================================================================================================================================================== #
# Obstacle Space Class Definition
# =========================================================================================================================================================================================================================== #
class ObstacleSpace:
    def __init__(self, width, height, res=1.0, clearance=30):
        self.res = res
        self.obstacle_space = np.ones((height, width, 3))
        self.robot_points = []
        self.obstacles = set()
        self.coordinates = set()
        self.circle_minowski = set()
        self.net_obstacle = set()
        self.minowski = set()
        self.clearance = int(res * clearance)

        t0 = time.time()

        # -----> Generate Robot Space <----- #
        print('Robot Point Creation: started..! \n')
        for y in range(-clearance, clearance + 1):
            for x in range(-clearance, clearance + 1):
                if x ** 2 + y ** 2 - clearance ** 2 <= 0:
                    self.robot_points.append((y, x))

        print('Robot Points Creation: Done..! \n')

        # -----> Generate Obstacle Space <----- #
        print('Obstacle Space Creation: started..! \n')
        for y in range(height):
            for x in range(width):
                self.rect_check(y, x)
                self.circle_check(y, x, 0)
                self.boundry_check(y, x)
                self.circle_check(y, x, 29)
                if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                    self.obstacle_space[y, x] = (0.6, 0.6, 0)
                    if (y, x) not in self.obstacles: self.obstacles.add((y, x))
                    if (y, x) not in self.coordinates: self.coordinates.add((y, x))

        t1 = time.time()

        print('Obstacle Creation: Done..! \n')
        print('TIme Elapsed: ', t1 - t0, '\n\n')

        # -----> Generate Minowski summed Obstacle Space <----- #
        print('Minowski sum Obstacle Creation: started..! \n')
        self.minowski_sum(self.robot_points, height, width)

        t2 = time.time()
        print('Minowski sum obstacle Creation: Done..! \n')
        print("Total TIme Elapsed: ", t2 - t0, '\n\n')

    # ======================================================================================================================================================================================================================= #
    # Function for rectange check
    # ======================================================================================================================================================================================================================= #
    def rect_check(self, y, x):
        if 150 <= x <= 309.7 and 100 <= y <= 260:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 01
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 832 <= x <= 918 and 0 <= y <= 183:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 02
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 983 <= x <= 1026 and 0 <= y <= 91:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 03
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 744 <= x <= 1111 and 313 <= y <= 389:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 04
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 1052 <= x <= 1111 and 444.5 <= y <= 561.5:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 05
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 1019 <= x <= 1111 and 561.5 <= y <= 647.5:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 06
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 784.5 <= x <= 934.5 and 626 <= y <= 743:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 07
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 529 <= x <= 712 and 669 <= y <= 745:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 08
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 438 <= x <= 529 and 512 <= y <= 695:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 09
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 1052 <= x <= 1111 and 714.8 <= y <= 831.8:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 10
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 927 <= x <= 1111 and 899 <= y <= 975:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 11
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 779 <= x <= 896 and 917 <= y <= 975:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 12
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 474 <= x <= 748 and 823 <= y <= 975:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 13
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))
        elif 685 <= x <= 1111 and 975 <= y <= 1011:
            self.obstacle_space[y, x] = [0.6, 0.6, 0]  # Rect-obstacle - 14
            if (y, x) not in self.obstacles:
                self.obstacles.add((y, x))

    # ======================================================================================================================================================================================================================= #
    # Function for Circle check
    # ======================================================================================================================================================================================================================= #
    def circle_check(self, y, x, flag=0):
        if flag:
            if (39.5) ** 2 < ((x - 390) ** 2 + (y - 45) ** 2) < (41.5) ** 2:
                self.coordinates.add((y, x))
            elif (39.5) ** 2 < ((x - 438) ** 2 + (y - 274) ** 2) < (41.5) ** 2:
                self.coordinates.add((y, x))
            elif (39.5) ** 2 < ((x - 438) ** 2 + (y - 736) ** 2) < (41.5) ** 2:
                self.coordinates.add((y, x))
            elif (39.5) ** 2 < ((x - 390) ** 2 + (y - 965) ** 2) < (41.5) ** 2:
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
    def boundry_check(self, y, x):
        if ((150 == x or x == 310) and 100 <= y <= 260) or (150 <= x <= 310 and (100 == y or y == 260)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 01
        if ((832 == x or x == 918) and 0 <= y <= 183) or (832 <= x <= 918 and (0 == y or y == 183)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 02
        if ((983 == x or x == 1026) and 0 <= y <= 91) or (983 <= x <= 1026 and (0 == y or y == 91)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 03
        if ((744 == x or x == 1111) and 313 <= y <= 389) or (744 <= x <= 1111 and (313 == y or y == 389)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 04
        if ((1052 == x or x == 1111) and 445 <= y <= 562) or (1052 <= x <= 1111 and (445 == y or y == 562)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 05
        if ((1019 == x or x == 1111) and 562 <= y <= 648) or (1019 <= x <= 1111 and (562 == y or y == 648)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 06
        if ((785 == x or x == 935) and 626 <= y <= 743) or (785 <= x <= 935 and (626 == y or y == 743)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 07
        if ((529 == x or x == 712) and 669 <= y <= 745) or (529 <= x <= 712 and (669 == y or y == 745)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 08
        if ((x == 438 or x == 529) and 512 <= y <= 695) or (438 <= x <= 529 and (y == 512 or y == 695)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 09
        if ((x == 1052 or x == 1111) and 715 <= y <= 832) or (1052 <= x <= 1111 and (y == 715 or y == 832)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 10
        if ((x == 927 or x == 1111) and 899 <= y <= 975) or (927 <= x <= 1111 and (y == 899 or y == 975)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 11
        if ((x == 779 or x == 896) and 917 <= y <= 975) or (779 <= x <= 896 and (y == 917 or y == 975)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 12
        if ((x == 474 or x == 748) and 823 <= y <= 975) or (474 <= x <= 748 and (y == 823 or y == 975)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 13
        if ((x == 685 or x == 1111) and 975 <= y <= 1011) or (685 <= x <= 1111 and (y == 975 or y == 1011)): self.coordinates.add((y, x))  # Boundry check for Rectangle: 14

    # ======================================================================================================================================================================================================================= #
    # Function for Minowski Sum
    # ======================================================================================================================================================================================================================= #
    def minowski_sum(self, robot_points, h, w):
        for y, x in self.coordinates:
            for j in robot_points:
                point = (y - (-1) * j[0], x - (-1) * j[1])
                if 0 <= point[1] < w and 0 <= point[0] < h:
                    if point not in self.obstacles:
                        self.minowski.add(point)
                        self.obstacle_space[point] = [0.6, 0.6, 0.6]

        self.net_obstacle = self.obstacles.union(self.minowski)


# =========================================================================================================================================================================================================================== #
# Path Finder Class Definition
# =========================================================================================================================================================================================================================== #
class AstarPathFinder:
    def __init__(self, start, goal, grid_size, bot_radius=22, clearance=8, resolution=1, vel=[5, 10], L=0, r=0):

        # ------> Important Variables <------- #
        self.start = start  # Start Co Ordinate of the Robot
        self.goal = goal  # End Co Ordinate for the Robot to reach
        self.res = resolution  # Resolution of the output
        self.grid_size = grid_size  # Height and Width of the Layout
        self.goal_area = list()  # Generating Gaol Space
        self.visited = set()  # Explored Nodes
        self.unvisited = list()  # Nodes yets to be explored in queue
        self.robot_points = []  # Contains coordinates of robot area
        self.obstacle_nodes = list()  # Given Obstacle space's nodes
        self.net_obstacles_nodes = set()  # New Obstacle space After Minowski Sum
        self.shortest_path = list()
        self.terminate = False

        # ------> Bot Variables <------- #
        self.bot_radius = bot_radius  # Radius of the Robot
        # Clearance of the robot to be maintained with obstacle + Robot's radius
        self.clearance = clearance + bot_radius
        self.R = r  # Radius of the Wheel
        self.L = L  # Wheel base of the bot
        self.vel = [(0, vel[0]), (0, vel[1]), (vel[0], 0), (vel[1], 0), (vel[1], vel[0]),
                    (vel[0], vel[1]), (vel[0], vel[0]), (vel[1], vel[1])]  # Velocity Combinations

        # ------> Environment Setup <------- #
        self.env = ObstacleSpace(grid_size[1], grid_size[0], self.clearance)
        self.net_obstacles_nodes = self.env.net_obstacle
        self.graph = self.env.obstacle_space.copy()  # GUI to vizualize the exploration
        self.path = self.env.obstacle_space.copy()
        # Generating nodes and initialising the Layout with given data
        self.nodes = self.generate_nodes(
            grid_size, start, goal, self.net_obstacles_nodes)
        # Calculating Robot occupied point cloud at origin
        self.calc_goal_area(10, self.goal)

    # ====================================================================================================================================================================================================================== #
    # -----> Function to generate Nodes <----- #
    # ====================================================================================================================================================================================================================== #
    def generate_nodes(self, grid_size, start, goal, obstacles):
        nodes = np.empty(grid_size, dtype=object)
        for y in range(grid_size[0]):
            for x in range(grid_size[1]):
                nodes[y, x] = Nodes((y, x), None, goal)
                if (y, x) in obstacles:
                    nodes[y, x].obstacle = True
        nodes[start] = Nodes(start, start, goal, 0.0)
        return nodes

    # ====================================================================================================================================================================================================================== #
    # -----> Function to Calculate Goal Space <----- #
    # ====================================================================================================================================================================================================================== #
    def calc_goal_area(self, clearance, goal):
        for y in range(goal[0] - clearance, clearance + goal[0] + 1):
            for x in range(goal[1] - clearance, clearance + goal[1] + 1):
                if ((goal[1] - x) ** 2 + (goal[0] - y) ** 2 - clearance ** 2) <= 0:
                    self.goal_area.append((y, x))
                    self.graph[y, x] = (0.1, 0.5, 0.1)
                    self.path[y, x] = (0.1, 0.5, 0.1)

    # ====================================================================================================================================================================================================================== #
    # -----> Function to Explore the neighbours <----- #
    # ====================================================================================================================================================================================================================== #
    def find_neighbours(self, current_node, dt=1):
        theta = self.nodes[current_node].theta
        R, L = self.R, self.L
        # Get the theta of the current node
        for vel in self.vel:
            ur, ul = vel
            
            dx = (dt * R * (ul + ur) * math.cos(theta)) / 2
            dy = (dt * R * (ul + ur) * math.sin(theta)) / 2
            d_theta = (dt * R * (ur - ul)) / L
            d = (dx ** 2 + dy ** 2) ** .5
            dx_ = d * math.cos(theta + d_theta)
            dy_ = d * math.sin(theta + d_theta)

            d_ = ((dx_) ** 2 + (dy_) ** 2) ** .5
            if (d_theta == 0): d_ = d
            else: d_ = ((3.14 - abs(d_theta)) * d) / (2 * math.sin(abs(d_theta)))

            # d_theta = (dt * R * (ur - ul)) / L
            # dx_ = (dt * R * (ul + ur) * math.cos(d_theta + theta)) / 2
            # dy_ = (dt * R * (ul + ur) * math.sin(d_theta + theta)) / 2
            # d_ = ((dx_) ** 2 + (dy_) ** 2) ** .5
            # if (d_theta == 0): continue
            # else: d_ = ((3.14 - abs(d_theta)) * d_) / (2 * math.sin(abs(d_theta)))
            
            lin_vel = d_
            new_node = int(current_node[0] + dy_), int(current_node[1] + dx_)
            if new_node in self.goal_area:
                print('D_Theta', d_theta, 'theta', theta)
            if 0 <= new_node[0] < 1011 and 0 <= new_node[1] < 1111:
                if new_node not in self.net_obstacles_nodes and new_node not in self.visited and not self.terminate:
                    self.compute_cost(new_node, current_node, d_, vel, lin_vel, d_theta, dt)
                    if self.nodes[new_node] not in self.unvisited:
                        self.unvisited.append(self.nodes[new_node])
                        cv.line(self.graph, (new_node[1], new_node[0]), (current_node[1], current_node[0]), (1, 0, 0), 1)
                        cv.circle(self.graph, (new_node[1], new_node[0]), 5, (0, 0, 1), 1)
                        cv.imshow("A* Algorithm", self.graph)
                        cv.waitKey(1)
                else:
                    continue
            if new_node in self.goal_area:
                print('D_Theta', d_theta, 'theta', theta)
                self.terminate = True

    # ====================================================================================================================================================================================================================== #
    # -----> Function to Calculate, compare, and Update the cost <----- #
    # ====================================================================================================================================================================================================================== #
    def compute_cost(self, node, parent, step_cost, velocity,lin_vel, d_theta, dt=1):
        initial_cost = self.nodes[parent].cost
        node_cost = self.nodes[node].cost
        if node_cost > initial_cost + step_cost:
            self.nodes[node].cost = initial_cost + step_cost
            self.nodes[node].parent = parent
            self.nodes[node].h_cost = self.nodes[node].heuristic_distance + self.nodes[node].cost
            self.nodes[node].velocity = velocity
            self.nodes[node].theta = self.nodes[parent].theta + d_theta
            self.nodes[node].omega = d_theta / dt
            self.nodes[node].linear_velocity = lin_vel / dt
        return self.nodes[node]

    # ====================================================================================================================================================================================================================== #
    # -----> A* Algorithm Function <----- #
    # ====================================================================================================================================================================================================================== #

    def Astar(self, start_index, goal_index):
        # Initialising the node to explore with start node
        self.unvisited.append(self.nodes[start_index])
        cv.namedWindow("A* Algorithm", cv.WINDOW_NORMAL)
        cv.circle(self.graph, (start_index[1], start_index[0]), 1, [ 255, 0, 255], -1)
        cv.circle(self.graph, (goal_index[1], goal_index[0]), 1, [0, 255, 0], -1)

        if self.goal not in self.net_obstacles_nodes and self.start not in self.net_obstacles_nodes:
            while self.unvisited:
                current_node = min(self.unvisited, key=lambda x: (x.h_cost, x.heuristic_distance))
                if current_node.index in self.goal_area: break
                self.find_neighbours(current_node.index)
                self.visited.add(current_node.index)
                self.unvisited.remove(current_node)
                self.graph[current_node.index] = (0, 1, 0)
                cv.circle(self.graph, (current_node.index[1], current_node.index[0]), 5, (0, 1, 0), 1)

            # -----> Back Tracking the node path <----- #
            child = current_node.index
            while True:
                parent = self.nodes[child].parent
                cv.line(self.path, (parent[1], parent[0]),(child[1], child[0]), (0, 1, 0), 1)
                cv.circle(self.path, (parent[1], parent[0]), 5, (0.6, 0.6, 0), 1)
                cv.circle(self.path, (parent[1], parent[0]), 1, (0, 0, 1), -1)
                if parent == child: break
                self.shortest_path.insert(0, self.nodes[child])
                print(parent)
                child = parent

            # # -----> To Skip to the result comment below lines <----- #  
            # for obj in self.shortest_path:
            #     print("point", obj.index, "lin Velocity:", obj.linear_velocity, "Ang:", obj.omega)
            print("cost in Astar: ", self.nodes[current_node.index].cost)
        else:
            print("The Goal node in obstacle or out of bound: ", self.goal in self.net_obstacles_nodes,
                  "\nThe start Node inside Obstacle or out of bound: ", self.start in self.net_obstacles_nodes)


def simulate(obj_list):
    print("simulation started")
    rospy.init_node('Planturtle', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()

    rate = rospy.Rate(10)
    vel_msg.linear.x = 0.0
    vel_msg.angular.z = 0.0
    n = 0
    c = 0
    while not rospy.is_shutdown():
    	if n < 61 :
            vel_msg.linear.x = obj_list[c].linear_velocity/600
            vel_msg.angular.z = -1*(obj_list[c].omega)/6
            print(c, n, vel_msg.linear.x, vel_msg.angular.z)
            n += 1
            
        if n==60:
            n= 0
            c += 1
        if(c == len(obj_list)):
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            pub.publish(vel_msg)
            break
        pub.publish(vel_msg)
        rate.sleep()
    # ====================================================================================================================================================================================================================== #

    # ====================================================================================================================================================================================================================== #
    # User Input Section
    # ====================================================================================================================================================================================================================== #


if __name__ == '__main__':
    try:
        rospy.sleep(10)
        sx = int(input("Enter the Start X value-just only number units in centimeters & origin at bottom-left: "))
        sy = int(input("Enter the Start Y value-just only number units in centimeters & origin at bottom-left: "))
        gx = int(input("Enter the Goal X value-just only number units in centimeters & origin at bottom-left: "))
        gy = int(input("Enter the Goal Y value-just only number units in centimeters & origin at bottom-left: "))

        start = (sx, sy)
        goal = (gx, gy)

        grid_size = (1111, 1011)
        bot_radius = 22
        clearance = 8
        res = 1
        vel = [5, 10]
        base_length = res * 28.7
        wheel_radius = res * 3.3

        start = (int(res * (grid_size[1] - start[1])), int(res * (start[0])) - 1)
        goal = (int(res * (grid_size[1] - goal[1])), int(res * (goal[0])) - 1)
        grid_size = (int(res * grid_size[1]), int(res * grid_size[0]))
        bot_radius = int(bot_radius * res)

        t0 = time.time()
        maps = AstarPathFinder(start, goal, grid_size, bot_radius,
                                clearance, res, vel, base_length, wheel_radius)
        maps.Astar(start, goal)
        t1 = time.time()
        simulate(maps.shortest_path)
        print("Total Algorithm run time: ", t1 - t0)

        # -----> Visualising the Graph <----- #
        cv.circle(maps.path, (start[1], start[0]), 6, (0.8, 0.8, 0), -1)
        cv.circle(maps.path, (start[1], start[0]), 6, (0, 0, 1), 2)
        cv.circle(maps.path, (goal[1], goal[0]), 10, (0, 0, 1), 2)
        cv.imshow("Shortest path: ", maps.path)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except rospy.ROSInterruptException:
        pass
# ====================================================================================================================================================================================================================== #
