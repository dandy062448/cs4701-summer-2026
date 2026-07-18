import numpy as numpy
from queue import PriorityQueue
from utils.utils import PathPlanMode, Heuristic, cost, expand, visualize_expanded, visualize_path
import numpy as np

def compute_heuristic(node, goal, heuristic: Heuristic):
    """ Computes an admissible heuristic value of node relative to goal.

    Args:
        node (tuple): The cell whose heuristic value we want to compute.
        goal (tuple): The goal cell.
        heuristic (Heuristic): The heuristic to use. Must
        specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.

    Returns:
        h (float): The heuristic value.

    """
    # TODO:

    return 0


def uninformed_search(grid, start, goal, mode: PathPlanMode):
    """ Find a path from start to goal in the gridworld using 
    BFS or DFS.

    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        mode (PathPlanMode): The search strategy to use. Must
        specify either PathPlanMode.DFS or PathPlanMode.BFS.

    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing
        the size of the frontier at each iteration.
    """

    frontier = [start]
    frontier_sizes = []
    expanded = []
    reached = {start: None}

    is_goal = False

    while frontier:
        frontier_sizes.append(len(frontier))

        if mode == PathPlanMode.DFS:
            # prioritize latest frontier nodes, which have higher depth.
            current = frontier.pop()
        elif mode == PathPlanMode.BFS:
            # prioritize earliest frontier nodes with lowest depth.
            current = frontier.pop(0)
        else:
            print("Error: invalid mode.")

        expanded.append(current)
        
        if current == goal:
            is_goal = True
            break
        
        for child in expand(grid, current):
            if child not in reached:
                frontier.append(child)
                reached[child] = current

    # Find goal path by starting at goal node and backstepping.
    path = []
    if is_goal:
        node = goal
        while node is not None:
            path.append(node)
            node = reached[node]
    path.reverse()
    
    return path, expanded, frontier_sizes


def a_star(grid, start, goal, mode: PathPlanMode, heuristic: Heuristic, width):
    """ Performs A* search or beam search to find the
    shortest path from start to goal in the gridworld.

    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.a
        mode (PathPlanMode): The search strategy to use. Must
        specify either PathPlanMode.A_STAR or
        PathPlanMode.BEAM_SEARCH.
        heuristic (Heuristic): The heuristic to use. Must
        specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.
        width (int): The width of the beam search. This should
        only be used if mode is PathPlanMode.BEAM_SEARCH.

    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing
        the size of the frontier at each iteration.
    """

    frontier = PriorityQueue()
    frontier.put((0, start))
    frontier_sizes = []
    expanded = []
    reached = {start: {"cost": cost(grid, start), "parent": None}}

    # TODO:
    path = []
    return path, expanded, frontier_sizes


def test_world(world_id, start, goal, h, width, animate, world_dir):
    print(f"Testing world {world_id}")
    grid = np.load(f"{world_dir}/world_{world_id}.npy")

    if h == 0:
        modes = [
            PathPlanMode.DFS,
            PathPlanMode.BFS
        ]
        print("Modes: 1. DFS, 2. BFS")
    elif h == 1 or h == 2:
        modes = [
            PathPlanMode.A_STAR,
            PathPlanMode.BEAM_SEARCH
        ]
        if h == 1:
            print("Modes: 1. A_STAR, 2. BEAM_A_STAR")
            print("Using Manhattan heuristic")
        else:
            print("Modes: 1. A_STAR, 2. BEAM_A_STAR")
            print("Using Euclidean heuristic")

    for mode in modes:

        search_type, path, expanded, frontier_size = None, [], [], []
        if mode == PathPlanMode.DFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "DFS"
        elif mode == PathPlanMode.BFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "BFS"
        elif mode == PathPlanMode.A_STAR:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, 0)
            search_type = "A_STAR"
        elif mode == PathPlanMode.BEAM_SEARCH:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, width)
            search_type = "BEAM_A_STAR"

        if search_type:
            print(f"Mode: {search_type}")
            path_cost = 0
            for c in path:
                path_cost += cost(grid, c)
            print(f"Path length: {len(path)}")
            print(f"Path cost: {path_cost}")
            if frontier_size:
                print(f"Number of expanded states: {len(frontier_size)}")
                print(f"Max frontier size: {max(frontier_size)}\n")
            if animate == 0 or animate == 1:
                visualize_expanded(grid, start, goal, expanded, path, animation=animate)
            else:
                visualize_path(grid, start, goal, path)