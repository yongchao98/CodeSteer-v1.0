def is_line_intersect_box(p1, p2, box):
    x_start, x_end, y_start, y_end = box

    def line_intersects_line(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    box_edges = [
        ((x_start, y_start), (x_end, y_start)),
        ((x_end, y_start), (x_end, y_end)),
        ((x_end, y_end), (x_start, y_end)),
        ((x_start, y_end), (x_start, y_start))
    ]

    for edge in box_edges:
        if line_intersects_line(p1, p2, edge[0], edge[1]):
            return True
    return False

def check_trajectory(box_dict, start_pos, end_pos, waypoints, instruct_index):
    # Define the environment
    black_walls = [
        (-1.5, -1.4, -1.5, 1.5), (1.4, 1.5, -1.5, 1.5),
        (-1.5, 1.5, -1.5, -1.4), (-1.5, 1.5, 1.4, 1.5)
    ]
    red_box = box_dict['red']
    blue_box = box_dict['blue']
    green_box = box_dict['green']
    yellow_box = box_dict['yellow']
    pink_box = box_dict['pink']
    purple_box = box_dict['purple']
    Orange_box = box_dict['orange']
    intersect_red = False
    intersect_blue = False
    intersect_green = False
    intersect_yellow = False
    intersect_pink = False
    intersect_purple = False
    intersect_orange = False

    if len(waypoints) < 2:
        return False, "Trajectory must have at least 2 waypoints."

    if waypoints[0] != start_pos:
        return False, "The first waypoint must be the starting position."

    if waypoints[-1] != end_pos:
        return False, "The last waypoint must be the ending position."

    # Validate the trajectory
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]

        # Check if the line intersects with any black walls or the blue box
        for wall in black_walls:
            if is_line_intersect_box(p1, p2, wall):
                return False, "Trajectory intersects with a black wall."

        if is_line_intersect_box(p1, p2, blue_box):
            intersect_blue = True
        if is_line_intersect_box(p1, p2, red_box):
            intersect_red = True
        if is_line_intersect_box(p1, p2, green_box):
            intersect_green = True
        if is_line_intersect_box(p1, p2, yellow_box):
            intersect_yellow = True
        if is_line_intersect_box(p1, p2, pink_box):
            intersect_pink = True
        if is_line_intersect_box(p1, p2, purple_box):
            intersect_purple = True
        if is_line_intersect_box(p1, p2, Orange_box):
            intersect_orange = True

    if instruct_index == 0:
      if not intersect_green:
        return False, "Trajectory does not enter the green box."
      if not intersect_yellow:
        return False, "Trajectory does not enter the yellow box."
    elif instruct_index == 1:
      if not intersect_yellow:
        return False, "Trajectory does not enter the yellow box."
      if not intersect_purple:
        return False, "Trajectory does not enter the purple box."
      if not intersect_red:
        return False, "Trajectory does not enter the red box."
      if not intersect_green:
        return False, "Trajectory does not enter the green box."
    elif instruct_index == 2:
      if not intersect_yellow:
        return False, "Trajectory does not enter the yellow box."
      if not intersect_purple:
        return False, "Trajectory does not enter the purple box."
      if not intersect_red:
        return False, "Trajectory does not enter the red box."
      if not intersect_green:
        return False, "Trajectory does not enter the green box."
      if intersect_blue:
        return False, "Trajectory wrongly enters the blue box."
    elif instruct_index == 3:
      if not intersect_green:
        return False, "Trajectory does not enter the green box."
      if not intersect_yellow:
        return False, "Trajectory does not enter the yellow box."
    elif instruct_index == 4:
      if not intersect_blue:
        return False, "Trajectory does not enter the blue box."
      if intersect_red or intersect_green or intersect_yellow or intersect_pink or intersect_purple or intersect_orange:
        return False, "Trajectory wrongly enters other colored boxes."
    elif instruct_index == 5:
      if not intersect_green:
        return False, "Trajectory does not enter the green box."
      if not intersect_yellow:
        return False, "Trajectory does not enter the yellow box."
      if intersect_pink:
        return False, "Trajectory wrongly enters the pink box."
    elif instruct_index == 6:
      if not intersect_yellow:
        return False, "Trajectory does not enter the yellow box."
      if not intersect_orange:
        return False, "Trajectory does not enter the orange box."
      if intersect_blue:
        return False, "Trajectory wrongly enters the blue box."
      if intersect_purple:
        return False, "Trajectory wrongly enters the purple box."
      if intersect_red:
        return False, "Trajectory wrongly enters the red box."
    elif instruct_index == 7:
      if not intersect_green:
        return False, "Trajectory does not enter the green box."
      if not intersect_yellow:
        return False, "Trajectory does not enter the yellow box."
      if not intersect_red:
        return False, "Trajectory does not enter the red box."
      if intersect_blue:
        return False, "Trajectory wrongly enters the blue box."
    elif instruct_index == 8:
      if not intersect_purple:
        return False, "Trajectory does not enter the purple box."
      if not intersect_orange:
        return False, "Trajectory does not enter the orange box."
      if not intersect_pink:
        return False, "Trajectory does not enter the pink box."
      if intersect_blue:
        return False, "Trajectory wrongly enters the blue box."
      if intersect_red:
        return False, "Trajectory wrongly enters the red box."
      if intersect_green:
        return False, "Trajectory wrongly enters the green box."
      if intersect_yellow:
        return False, "Trajectory wrongly enters the yellow box."
    elif instruct_index == 9:
      if not intersect_yellow:
        return False, "Trajectory does not enter the yellow box."
      if not intersect_red:
        return False, "Trajectory does not enter the red box."
      if not intersect_green:
        return False, "Trajectory does not enter the green box."
      if intersect_blue:
        return False, "Trajectory wrongly enters the blue box."

    return True, "Trajectory is valid."
