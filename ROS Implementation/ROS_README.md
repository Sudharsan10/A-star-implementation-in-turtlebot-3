## Instruction to Run ROS Implementation

- Unzip the "catkin_ws" folder, set your terminal to the extracted folder.
- The algorith.py with ROS implementation code could be found in the folder 
    > ```
    > /Catkin_ws/src/astar/scripts/Algorithm.py’
    > ```

- To run the simulation in gazebo, navigate to the root folder and launch terminal and run the following command 
    > ```
    > $ catkin_make
    > 
    > $ source devel/setup.bash
    > 
    > $ export TURTLEBOT3_MODEL=waffle_pi
    > 
    > $ roslaunch astar astar.launch x_pos:=800 y_pos:=600
    > ```

- If (start_x, start_y) are the start nodes of the robot given in cms, then (Xg, Yg) = (start_X – 555), (start_y - 505)

- Follow the on screen instructions to enter the start node,  goal node. Note that the start and goal nodes are in
 map’s coordinate (x,y) and the grid layout is (1111,1011 and are entered in centimeters. The program starts with 
 running the A* algorithm, pops the layout and exploration and once it finds the shortest path to the goal area with a 
 tolerance of 10 cm around the goal node.
 
- After successfully finding the shortest path you could notice that the turtlebot inside the simulation starts 
tracking the same path we estimated via A* algorithm

**Note:** Gazebo co ordinates are in meters and center of the map is the origin, whereas the A* algorithm takes start 
and end node as (x,y) in centimeters with bottom left as origin.