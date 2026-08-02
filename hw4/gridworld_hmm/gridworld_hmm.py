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

    def initT(self):
        """
        Create and return NxN transition matrix, where N = size of grid.
        """
        # TODO
        return np.ones((self.grid.size, self.grid.size)) / self.grid.size

    def initO(self):
        """
        Create and return 16xN matrix of observation probabilities, where N = size of grid.
        """
        # TODO
        return np.ones((16, self.grid.size)) / 16


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