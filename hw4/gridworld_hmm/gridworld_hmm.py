import numpy as np
import numpy.typing as npt

class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: list = [], num_particles: int = 30):
        if walls:
            self.grid = np.ones(size)
            for cell in walls:
                self.grid[cell] = 0
        else:
            self.grid = np.random.randint(2, size=size)

        self.init = (self.grid / np.sum(self.grid)).flatten()

        self.epsilon = epsilon
        self.particles = np.random.choice(len(self.init), size=num_particles, p=self.init)
        self.weights = np.ones(num_particles)

        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        m, n = self.grid.shape
        adjacent = [(i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]
        neighbors = [(i, j)]
        for a1, a2 in adjacent:
            if 0 <= a1 < m and 0 <= a2 < n and self.grid[a1, a2] == 1:
                neighbors.append((a1, a2))
        return neighbors


    """
    4.1 and 4.2. Transition and observation probabilities
    """

    def row_to_grid_index(self, row_index):
        """
        Helper function that returns the corresponding grid coordinates 
        given a row index of an NxN transition matrix.
        """
        _, row_size = self.grid.shape
        return [row_index // row_size, row_index % row_size]


    def grid_to_row_index(self, grid_tuple):
        """
        Helper function that returns the corresponding row index of an NxN transition matrix 
        given a grid coordinate.
        """
        _, row_size = self.grid.shape
        return grid_tuple[0] * row_size + grid_tuple[1]


    @staticmethod
    def get_adjacent(cell):
        """
        Helper function that returns cells to the north, east, south, and west of the 
        given cell respectively.
        """
        i, j = cell
        return [(i - 1, j), (i, j + 1), (i + 1, j),(i, j - 1)]


    def initT(self):
        """
        Create and return NxN transition matrix, where N = size of grid.
        """
        print(self.grid)
        grid = np.zeros((self.grid.size, self.grid.size))

        for row_index in range(self.grid.size):
            grid_element = self.row_to_grid_index(row_index)
            neighbors = self.neighbors(grid_element)
            # print(f"neighbors of {grid_element}: {neighbors}")

            for neighbor in neighbors:
                neighbor_index = self.grid_to_row_index(neighbor)
                grid[row_index][neighbor_index] = 1 / len(neighbors)

        # print(grid)
        return grid


    def initO(self):
        """
        Create and return 16xN matrix of observation probabilities, where N = size of grid.
        """
        correct_observations = []

        for row_index in range(self.grid.size):
            grid_element = self.row_to_grid_index(row_index)
            neighbors = self.neighbors(grid_element)
            current_cell = neighbors[0]
            # print(f"current cell: {current_cell}")

            observation_binary = ""
            adjacent_cells = Gridworld_HMM.get_adjacent(current_cell)
            # print(f"adjacent cells: {adjacent_cells}")
            for cell in adjacent_cells:
                if cell in neighbors:
                    observation_binary += "1"
                else:
                    observation_binary += "0"

            observation = int(observation_binary, 2)
            correct_observations.append(observation)

        observation_probabilities = np.zeros((16, self.grid.size))
        for i in range(16):
            for j in range(self.grid.size):
                discrepancy = bin(i ^ correct_observations[j]).count('1')
                observation_probabilities[i][j] = \
                    self.epsilon ** discrepancy * (1 - self.epsilon) ** (4 - discrepancy)

        # sanity check: columns add to 1
        # for i in range(self.grid.size):
            # print(sum(observation_probabilities[:, i]))
        return observation_probabilities


    """
    4.3. Forward algorithm
    """

    def forward(self, observations: list[int]):
        """Perform forward algorithm over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Estimated belief state at each timestep.
        """
        # TODO
        return np.zeros((len(observations), self.grid.size))


    """
    4.4. Particle filter
    """

    def transition(self):
        """
        Sample the transition matrix for all particles.
        Update self.particles in place.
        """
        # TODO
        pass

    def observe(self, observation):
        """
        Compute the weights for all particles.
        Update self.weights in place.
        Args:
          obs (int): Integer observation value.
        """
        # TODO
        pass

    def resample(self):
        """
        Resample all particles.
        Update self.particles and self.weights in place.
        """
        # TODO
        pass

    def particle_filter(self, observations: list[int]):
        """Apply particle filter over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Counts of particles in each state at each timestep.
        """
        # TODO
        return np.zeros((len(observations), self.grid.size))