import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product
import time
from matplotlib.patches import Rectangle
from Attend2Pack2 import Attend2Pack
import torch

class BoxStackingEnv(gym.Env):
    fig = plt.figure()

    def __init__(self, grid_size=(20, 20), num_boxes=20, max_height=20, mode="human"):
        super(BoxStackingEnv, self).__init__()

        # Parameters
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.max_height = max_height
        self.mode = mode

        # Action space: discrete, representing the index of the box to select
        self.action_space = spaces.Discrete(num_boxes)

        # Observation space: 2D height map with values between 0 and max_height
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=grid_size, dtype=np.int32)

        # Initialize the height map
        self.height_map = np.zeros(self.grid_size, dtype=np.int32)

        # Initialize the set of 3D boxes (random heights for this example)
        self.boxes = self.create_boxes(grid_size, max_height, num_boxes)
        self.sorted_boxes = []
        self.current_step = 0

    def reset(self, seed=0):
        np.random.seed(seed)
        self.height_map = np.zeros(self.grid_size, dtype=np.int32)
        self.boxes = self.create_boxes(self.grid_size, self.max_height, self.num_boxes)
        self.sorted_boxes = []
        self.current_step = 0
        info = {}
        return self.height_map, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        box = self.boxes[action]
        success = self.placement_policy(box)
        truncated = not success
        self.current_step += 1
        done = self.current_step >= self.num_boxes or not success


        if not success or done:
            reward = self.compute_reward()
        else:
            reward = 0
        # Example reward: negative of the variance in heights (encourages even stacking)
        # reward = -np.var(self.height_map)

        return self.height_map, reward, done, truncated, {"sorted_boxes": self.sorted_boxes}

    def render(self):
        if self.mode == 'print':
            print("Height Map:")
            print(self.height_map)
        elif self.mode == 'human_verbose':
            self.plot_height_map(self.height_map)
            time.sleep(0.5)

    def close(self):
        pass

    def compute_reward(self):
        missed_volume = (sum(box[0] * box[1] * box[2] for box in self.boxes) -
                         sum(box[0] * box[1] * box[2] for box in self.sorted_boxes))  # Calculate volume of missed boxes
        total_volume = self.height_map.sum()
        reward = (total_volume - missed_volume) / (
                    self.height_map.max() * self.height_map.shape[0] * self.height_map.shape[1])
        return reward
    def placement_policy(self, box):
        width, depth, height = box

        # create all possible case rotations
        case_rotations = [[width, depth, height],
                          [width, height, depth],
                          [height, width, depth],
                          [height, depth, width],
                          [depth, height, width],
                          [depth, width, height]]
        # create list of possible valid locations
        potential_locations = []

        for i, case_rotation in enumerate(case_rotations):
            location = self.find_fit_location(case_rotation[:2])
            if location:
                location[-1] += case_rotations[i][-1]
                potential_locations.append((i, location))

        sorted_locations = sorted(potential_locations, key=lambda x: (x[1][2], x[1][1], x[1][0]))

        if sorted_locations:
            x, y, z = sorted_locations[0][1]
            width, depth, height = case_rotations[sorted_locations[0][0]]
            z -= height
            box = (width, depth, height, x, y, z)
            self.sorted_boxes.append(box)

            self.update_height_map(box)
            return True
        else:
            return False

    def update_height_map(self, box):
        # find the min and max of the top vertices
        min_x, min_y = (box[3], box[4])
        max_x, max_y = (min_x + box[0], min_y + box[1])
        height = box[2]
        self.height_map[min_x:max_x, min_y:max_y] += height
    def find_fit_location(self, block_size):
        block_height, block_width = block_size
        rows, cols = self.height_map.shape

        # Shape and strides for the view with overlapping blocks
        shape = (rows - block_height, cols - block_width, block_height, block_width)
        strides = self.height_map.strides + self.height_map.strides
        # neg_shape = len([x for x in list(shape) if x < 0]) > 0
        # neg_strides = len([x for x in list(strides) if x < 0]) > 0

        # Create a view into the frontier with the specified block shape
        blocks = as_strided(self.height_map, shape=shape, strides=strides)

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

    @staticmethod
    def create_boxes(grid_size, max_height, num_boxes):
        boxes = []
        max_size = grid_size[0] // 2
        for _ in range(num_boxes): # Generate dimensions for each box
            dims = []
            for dim in (*grid_size, grid_size[0]):
                possible_sizes = sorted(set([dim // i for i in range(2, max_size) if dim // i >= 1]))  # Create a list of unique, valid sizes
                if possible_sizes:  # Check if there are any valid sizes
                    dims.append(np.random.choice(possible_sizes))  # Sample a size from this list
                else:
                    dims.append(1)  # If no valid sizes, use 1 as the minimum size
            boxes.append(dims)
        return np.array(boxes)

    @staticmethod
    def plot_boxes_3d(boxes, custom_gap=0.05):

        def get_vertices(box, custom_gap):
            return [(box[3] + dx * (box[0] - custom_gap), box[4] + dy * (box[1] - custom_gap),
                     box[5] + dz * (box[2] - custom_gap)) for dz, dy, dx in product((0, 1), repeat=3)]

        def centroid(box):
            return (box[3] + box[0] / 2, box[4] + box[1] / 2, box[5] + box[2] / 2)  #centroid of box

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['cyan', 'magenta', 'yellow', 'green']
        maxVals, minVals = np.array([-np.inf, -np.inf, -np.inf]), np.array([np.inf, np.inf, np.inf])
        custom_gap = 0.5

        for i, box in enumerate(boxes):
            vertices = get_vertices(box, custom_gap)

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

            ax.text(*centroid(box), f'{caseNum}', color='black', fontsize=12, ha='center',
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

        plt.show()

    @staticmethod
    def plot_height_map(height_map):
        """
        Plots a 2D array as a height map with a highlighted region.

        Parameters:
        - data: 2D numpy array representing the height map.
        - input_position: Tuple (x, y) representing the starting position to highlight.
        - width: Width of the rectangle to highlight.
        - depth: Depth (height) of the rectangle to highlight.
        """
        fig, ax = plt.subplots()
        cax = ax.matshow(height_map.T, cmap='Pastel1')
        fig.suptitle("Height Map")

        # Annotate each cell with the value of the array, rounded to integers
        for (i, j), val in np.ndenumerate(height_map.T):
            ax.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=6)  # Display as integer


        # Flip the x-axis
        ax.set_xlim(height_map.shape[1] - 0.5, -0.5)  # Set the x-axis limits in reverse

        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Adjust ticks and labels
        ax.set_xticks(np.arange(height_map.shape[1]))
        ax.set_yticks(np.arange(height_map.shape[0]))
        ax.set_xticklabels(np.arange(height_map.shape[1]))  # Tick labels are in the original order now
        ax.set_yticklabels(np.arange(height_map.shape[0]))
        plt.show()

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().to("cpu"))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

def policy_gradient_loss(log_probs, model_rewards):
    loss = -1 * model_rewards * sum(log_probs)
    return loss

# Usage example
if __name__ == "__main__":
    import torch
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # General Parameters and Hyperparameters
    bin_size = (30, 30, 30)
    input_size = 3  # Adjust based on our box representation
    weight_decay = 0
    num_heads = 8
    amsgrad = False
    plot_results = False
    custom_gap = 0.05
    alpha = 0.05
    split = 0.8
    num_attn_layers = 3
    dropout = 0.0

    # Most important parameters
    num_boxes = 20
    learning_rate = 1e-4
    embedding_dim = 128
    hidden_dim = 512

    env = BoxStackingEnv(grid_size=(bin_size[0], bin_size[1]), num_boxes=20, max_height=50, mode="human")
    model = Attend2Pack(input_size=input_size,
                        num_cases=num_boxes,
                        device=device,
                        embedding_dim=embedding_dim,
                        num_heads=num_heads,
                        hidden_dim=hidden_dim,
                        num_attn_layers=num_attn_layers,
                        dropout=dropout,
                        bin_size=bin_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay,
                                  amsgrad=amsgrad)
    prev_reward = 0
    for episode in range(1000):
        obs, _ = env.reset()
        mask = np.ones(env.num_boxes, dtype=np.int8)
        log_probs = []
        reward = 0
        for _ in range(env.num_boxes):
            # unselected_boxes = env.boxes[mask == 1]
            action, log_prob, entropy = model.get_action_and_value(env.boxes, mask, obs)
            log_probs.append(log_prob)
            mask[action] = 0
            obs, reward, done, truncated, info = env.step(action)
            env.render()

            if not truncated:
                box = env.sorted_boxes[-1]
                if(env.mode == "human_verbose"):
                    print(f"selected box {action}, with: w:{box[0]}, d:{box[1]}, h:{box[2]}")
            else:
                box = env.boxes[action]
                print(f"failed to place {action}, with: w:{box[0]}, d:{box[1]}, h:{box[2]}")
            if done or truncated:
                print("reward: ", reward)
                prev_reward = reward

                if(episode % 50 == 0 and env.mode == "human" or env.mode == "human_verbose"):
                    env.plot_boxes_3d(env.sorted_boxes, custom_gap=0.01)
                    time.sleep(0.25)
                break
        #  TODO: torch stack logprobs instead of list
        loss = policy_gradient_loss(log_probs, reward)
        writer.add_scalar("loss", loss, episode)
        writer.add_scalar("reward", reward, episode)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(episode % 50 == 0 and env.mode == "human" or env.mode == "human_verbose"):
            plot_grad_flow(model.named_parameters())
            time.sleep(0.25)


    # env.close()
    writer.flush()
