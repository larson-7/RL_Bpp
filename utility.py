import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
import unittest
from itertools import product

class Case:
    fig = plt.figure()
    n_dims = 4  # number of input features for model to consider; width, height, depth, weight
    def __init__(self, width, depth, height, weight, x=0, y=0, z=0, com=None, id=0):
        self.width, self.height, self.depth, self.weight = width, height, depth, weight
        self.x, self.y, self.z = x, y, z
        self.id = id
      
    @property
    def com(self):
        return (self.x + self.width / 2, self.y + self.depth / 2, self.z + self.height / 2) # center of mass (com)
    
    @staticmethod
    def generate_random_cases(numcases, bin_size=(10, 10, 10), random_flag=True):
        cases = []
        if random_flag:
            for _ in range(numcases): # Generate dimensions for each case
                dims = []
                for dim in bin_size:
                    possible_sizes = sorted(set([dim // i for i in range(2, 11) if dim // i >= 1]))  # Create a list of unique, valid sizes
                    if possible_sizes:  # Check if there are any valid sizes
                        dims.append(np.random.choice(possible_sizes))  # Sample a size from this list
                    else:
                        dims.append(1)  # If no valid sizes, use 1 as the minimum size

                cases.append(Case(width=dims[0], height=dims[1], depth=dims[2], weight=np.random.uniform(0.5, 5.0), id=_)) # Create a new Case with these dims and weight
        else:
            case_sizes = [
                [2, 2, 2, 0.5],  # Tiny item
                [4, 4, 4, 1.0],  # Small item
                [5, 5, 5, 2.0],  # Small item
                [5, 2, 5, 2.0],  # Small item
                [5, 5, 2, 2.0],  # Small item
                [10, 5, 10, 2.5],  # Medium item
                [10, 10, 5, 2.5],  # Medium item
                [10, 5, 5, 2.5],  # Medium item
                [10, 10, 10, 2.5],  # Medium item
                [15, 10, 10, 2.5],  # Medium item
                [15, 15, 10, 2.5],  # Medium item
                [15, 5, 10, 2.5],  # Medium item
                # [20, 20, 20, 10], # Large item
                # [14, 4, 1, 0.5],  # Flat or slim item
                # [15, 15, 1, 3.0]  # Flat or slim item
            ]

            random_indices = np.random.choice(len(case_sizes), size=numcases, replace=True)
            cases = []

            for choice_idx in random_indices:
                caseDim = case_sizes[choice_idx]
                cases.append(Case(width=caseDim[0], height=caseDim[1], depth=caseDim[2], weight=caseDim[3]))

        return cases
    
    def get_vertices(self):
        return [(self.x + dx * (self.width - 1), self.y + dy * (self.depth - 1), self.z + dz * (self.height - 1)) for dz, dy, dx in product((0, 1), repeat=3)]
    
    def get_vertices_2(self, custom_gap):
        return [(self.x + dx * (self.width - custom_gap), self.y + dy * (self.depth - custom_gap), self.z + dz * (self.height - custom_gap)) for dz, dy, dx in product((0, 1), repeat=3)]    

    @property
    def position(self):
        return(self.x, self.y, self.z)

    @property
    def to_tensor(self):
        return torch.Tensor([self.width, self.height, self.depth, self.weight])

    @staticmethod
    def plot_cases(cases, custom_gap=0.05):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['cyan', 'magenta', 'yellow', 'green']
        maxVals, minVals = np.array([-np.inf, -np.inf, -np.inf]), np.array([np.inf, np.inf, np.inf])
        custom_gap = 0.5

        for i, case in enumerate(cases):
            vertices = case.get_vertices_2(custom_gap)

            # find the min/max x,y,z values of the cases
            localMaxVals = np.max(vertices, axis=0)
            localMinVals = np.min(vertices, axis=0)
            maxVals = np.maximum(localMaxVals, maxVals)
            minVals = np.minimum(localMinVals, minVals)

            verts = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front surface
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back surface
                [vertices[0], vertices[1], vertices[3], vertices[2]],  # Bottom surface
                [vertices[4], vertices[5], vertices[7], vertices[6]],  # Top surface
                [vertices[0], vertices[2], vertices[6], vertices[4]],  # Left surface
                [vertices[1], vertices[3], vertices[7], vertices[5]]  # Right surface
            ]

            color = colors[i % len(colors)]
            caseNum = i + 1
            ax.add_collection3d(Poly3DCollection(verts,
                                                facecolors=color,
                                                linewidths=1,
                                                edgecolors='black',
                                                alpha=.25))

            ax.text(*case.com, f'{caseNum}', color='black', fontsize=12, ha='center',
                    va='center')  # Labeling the case and including center of mass

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Adjust limits
        range_vals = maxVals - minVals
        maxVals += range_vals * 0.1
        minVals -= range_vals * 0.1
        
        ax.set_xlim([minVals[0], maxVals[0]])
        ax.set_ylim([minVals[1], maxVals[1]])
        ax.set_zlim([minVals[2], maxVals[2]])
        
        # Set a fixed view
        ax.view_init(elev=20, azim=45)
        
        # Ensure aspect ratio is equal
        ax.set_box_aspect((np.ptp(ax.get_xlim()),
                        np.ptp(ax.get_ylim()),
                        np.ptp(ax.get_zlim())))

        return plt

class PotentialCaseLocation:
    def __init__(self, origin_x, origin_y, x_dim, y_dim, z_height):
        self.origin_x, self.origin_y = origin_x, origin_y
        self.x_dim, self.y_dim, self.z_height = x_dim, y_dim, z_height

class StaticPlacementPolicy:
    # Finds the next available space given case orientation.
    def __init__(self, bin_size=(10,10,10), device='cuda'):
        self.bin_x_size, self.bin_y_size, self.bin_z_size = bin_size[0], bin_size[1], bin_size[2]    # bin size, x-dimension and y-dimension and z-dimension
        self.device = device
        self.frontier = np.zeros((self.bin_x_size, self.bin_y_size), dtype=np.float32)

    def reset(self):
        self.frontier = np.zeros((self.bin_x_size, self.bin_y_size), dtype=np.float32)

    def update_frontier_height_map(self, case):
        # create an xy view of the highest objects that are on the pallet
        top_vertices = case.get_vertices()[-4:]
        # find the min and max of the top vertices
        min_x, max_x, min_y, max_y = min([v[0] for v in top_vertices]), max([v[0] for v in top_vertices]), min([v[1] for v in top_vertices]), max([v[1] for v in top_vertices])
        height = int(top_vertices[0][2]) + 1

        self.frontier[min_x:max_x + 1, min_y:max_y + 1] = height

    def find_fit_location(self, block_size):
        block_height, block_width = block_size
        rows, cols = self.frontier.shape

        # Shape and strides for the view with overlapping blocks
        shape = (rows - block_height + 1, cols - block_width + 1, block_height, block_width)
        strides = self.frontier.strides + self.frontier.strides

        # Create a view into the frontier with the specified block shape
        blocks = as_strided(self.frontier, shape=shape, strides=strides)

        # Check uniformity: compare each block to its top-left element
        uniform_blocks = np.all(blocks == blocks[:, :, 0, 0, np.newaxis, np.newaxis], axis=(2, 3))

        # Get the indices and values of uniform blocks
        uniform_locations = np.argwhere(uniform_blocks)

        if uniform_locations.size == 0:
            return []  # No uniform block found

        # Get the top-left value of each uniform block
        block_values = blocks[uniform_blocks, 0, 0]

        # Find the index of the minimum value block
        min_z_value = np.min(block_values)
        min_value_indices = np.where(block_values == min_z_value)
        min_value_locations = uniform_locations[min_value_indices]

        # Get the indices that would sort the array by y then x
        sorted_indices = np.lexsort((min_value_locations[:, 0], min_value_locations[:, 1]))

        # Apply the indices to sort the array
        sorted_locations = min_value_locations[sorted_indices]

        # Return the location of the minimum value block
        return [*sorted_locations[0].tolist(), int(min_z_value)]

    def get_action(self, next_case: Case):
        # create all possible case rotations
        case_rotations = [[next_case.width, next_case.depth, next_case.height],
                          [next_case.width, next_case.height, next_case.depth],
                          [next_case.height, next_case.width, next_case.depth],
                          [next_case.height, next_case.depth, next_case.width],
                          [next_case.depth, next_case.height, next_case.width],
                          [next_case.depth, next_case.width, next_case.height]]
        # create list of possible valid locations
        potential_locations = []

        for i, case_rotation in enumerate(case_rotations):
            location = self.find_fit_location(case_rotation[:2])
            if location:
                location[-1] += case_rotations[i][-1]
                potential_locations.append((i, location))
        
        sorted_locations = sorted(potential_locations, key=lambda x: (x[1][2], x[1][1], x[1][0]))



        if sorted_locations:
            next_case.x, next_case.y, next_case.z = sorted_locations[0][1]
            next_case.width, next_case.depth, next_case.height = case_rotations[sorted_locations[0][0]]
            next_case.z -= next_case.height
            self.update_frontier_height_map(next_case)
            return True
        else:
            return False
        
        # # find next space in pallet given input dimensions, favor lowest Z location
        # valid_location = self.find_fit_location((next_case.width, next_case.depth))
        # if valid_location:
        #     next_case.x, next_case.y, next_case.z = valid_location[0], valid_location[1], int(self.frontier[valid_location[0], valid_location[1]])
        #     self.update_frontier_height_map(next_case)
        #     return True
        # else:
        #     return False


class TestStaticPlacementPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = StaticPlacementPolicy(bin_size=(10, 20, 10), device='cpu')

    def test_case_1(self):
        # Test placing the first case in an empty bin
        new_case = Case(2, 2, 2, 1.0)
        success = self.policy.get_action(new_case)
        self.assertTrue(success)
        self.assertEqual(new_case.position, (0, 0, 0))  # Should place at the origin

    def test_case_2(self):
        # Test placing a case next to an existing one
        cases = [Case(2, 2, 2, 1.0)]
        for case in cases:
            self.policy.get_action(case)
        new_case = Case(2, 2, 2, 1.0)
        success = self.policy.get_action(new_case)
        self.assertTrue(success)
        self.assertEqual(new_case.position, (2, 0, 0))  # Should place next to the existing case

    def test_case_3(self):
        # Test stacking a case on top of another
        cases = [Case(5, 10, 5, 1.0)]
        for case in cases:
            self.policy.get_action(case)
        first_case = Case(5, 10, 5, 1.0)
        success = self.policy.get_action(first_case)
        self.assertTrue(success)
        self.assertEqual(first_case.position, (5, 0, 0)) # Should place next to the first case

        second_case = Case(5, 10, 5, 1.0)
        success = self.policy.get_action(second_case)
        self.assertTrue(success)
        self.assertEqual(second_case.position, (0, 0, 5)) # Should place on top of the first case

    def test_case_4(self):
        # Test placing cases in a more complex arrangement
        cases = [
            Case(2, 2, 2, 1.0),
            Case(2, 2, 2, 1.0),
            Case(2, 2, 2, 1.0)
        ]
        for case in cases:
            self.policy.get_action(case)
        new_case = Case(2, 2, 2, 1.0)
        success = self.policy.get_action(new_case)
        self.assertTrue(success)
        self.assertEqual(new_case.position, (6, 0, 0))  # Should place in the next available slot

    def test_case_5(self):
        # Test when there is no space available for the new case
        cur_policy = StaticPlacementPolicy(bin_size=(4, 4, 4), device='cpu')
        cases = [
            Case(2, 2, 2, 1.0),
            Case(2, 2, 3, 1.0),
            Case(2, 2, 4, 1.0),
            Case(2, 2, 5, 1.0)
        ]
        for case in cases:
            cur_policy.get_action(case)
        new_case = Case(3, 3, 3, 2.0)
        success = cur_policy.get_action(new_case)
        self.assertFalse(success) # No space available for the new case

    def test_case_6(self):
        # Test when there is no space available for the new case
        cur_policy = StaticPlacementPolicy(bin_size=(4, 4, 4), device='cuda')
        cases = [
            Case(2, 2, 2, 1.0),
            Case(2, 2, 2, 1.0),
            Case(2, 2, 2, 1.0),
            Case(2, 2, 2, 1.0)
        ]
        for case in cases:
            cur_policy.get_action(case)
        new_case = Case(5, 4, 5, 2.0)
        success = cur_policy.get_action(new_case)
        self.assertFalse(success) # No space available for the new case

        new_case = Case(4, 5, 5, 2.0)
        success = cur_policy.get_action(new_case)
        self.assertFalse(success) # No space available for the new case

    def test_case_7(self):
        # Test when there is no space available for the new case
        cur_policy = StaticPlacementPolicy(bin_size=(10, 10, 10), device='cpu')
        cases = [
            Case(4, 4, 4, 1.0),
            Case(2, 2, 2, 1.0)
        ]
        for case in cases:
            cur_policy.get_action(case)
        new_case = Case(8, 8, 8, 2.0)
        success = cur_policy.get_action(new_case)
        self.assertFalse(success) # No space available for the new case

    def test_case_8(self):
        # Test when there is no space available for the new case
        cur_policy = StaticPlacementPolicy(bin_size=(10, 10, 10), device='cpu')
        cases = [
            Case(2, 2, 2, 1.0),
            Case(10, 2, 3, 1.0),
        ]
        for case in cases:
            cur_policy.get_action(case)
        new_case = Case(2, 2, 4, 1.0)
        success = cur_policy.get_action(new_case)
        # print(" ")
        # print(cur_policy.frontier)
        self.assertTrue(success)
        self.assertEqual(new_case.position, (2, 0, 0))  # Should go back in placement


if __name__ == '__main__':
    unittest.main()