import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import gym
import pacman_gym
from pacman_gym.envs.goal_finding import sample_layout
import csv
from pls.dpl_policies.goal_finding.util import get_ground_wall
import torch as th
import pandas as pd
import numpy as np
from skimage import io
from matplotlib import pyplot as plt

WALL_COLOR = 0.25
GHOST_COLOR = 0.5
PACMAN_COLOR = 0.75
FOOD_COLOR = 1
from skimage.measure import block_reduce

def generate_random_images(csv_path, folder, n_images=10):
    f_csv = open(csv_path, "w")
    writer = csv.writer(f_csv)
    writer.writerow(["image_name", "ghost(up)", "ghost(down)", "ghost(left)", "ghost(right)"])
    config = {
        "env_type": "GoalFinding-v0",
        "env_features":{
            "layout": "grid7x7_5_ghosts",
            "reward_goal": 10,
            "reward_crash": 0,
            "reward_food": 0,
            "reward_time": -0.1,
            "render": False,
            "max_steps": 200,
            "num_maps": 1,
            "seed": 567,
            'render_mode': "gray",
            'height': 240,
            'width': 240,
            'downsampling_size': 1
        }
    }
    env_name = config["env_type"]
    env_args = config["env_features"]
    env = gym.make(env_name, **env_args)
    env.env.gameDisplay = env.env.display
    env.env.rules.quiet = False

    for n in range(n_images):
        layout = sample_layout(
            env.layout.width,
            env.layout.height,
            env.env.num_agents,
            env.env.num_food,
            env.env.non_wall_positions,
            env.env.wall_positions,
            env.env.all_edges)
        env.env.game = env.rules.newGame(
            layout,
            env.env.pacman,
            env.env.ghosts,
            env.env.gameDisplay,
            env.env.beQuiet,
            env.env.catchExceptions,
            env.env.symX,
            env.env.symY,
        )
        env.game.start_game()
        env.env.render()


        img = env.game.compose_img("rgb")
        path = os.path.join(folder, f"img{n:06}.jpeg")
        plt.imsave(path, img)

        tinyGrid = env.game.compose_img("tinygrid")
        tinyGrid = th.tensor(tinyGrid).unsqueeze(0)
        ground_truth_ghost = get_ground_wall(tinyGrid, PACMAN_COLOR, GHOST_COLOR)
        row = [f"img{n:06}.jpeg"] + ground_truth_ghost.flatten().tolist()
        writer.writerow(row)
        f_csv.flush()
        if n % 10 == 0:
            print('Produce: {}/{} [({:.0f}%)]'.format(n, n_images, float(n)/n_images))

    f_csv.close()

class Goal_Finding_Dataset(Dataset):
    """Goal Finding dataset."""
    def __init__(self, csv_file, root_dir, downsampling_size,
                 train=False,
                 transform=None, n_train=400, n_test=100):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.n_train = n_train
        self.n_test = n_test
        self.downsampling_size = downsampling_size
        if train:
            self.instances = pd.read_csv(csv_file)[:self.n_train]
        else:
            self.instances = pd.read_csv(csv_file)[self.n_train: self.n_train + self.n_test]
        self.root_dir = root_dir
        self.transform = transform
        # self.use_grayscale = use_grayscale
        self.train = train

    def __len__(self):
        return len(self.instances)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [-1, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            gray = gray / 128. - 1.
        return th.tensor(gray,dtype=th.float32)

    @staticmethod
    def downsampling(x, downsampling_size):
        dz = block_reduce(x.squeeze(), block_size=(downsampling_size, downsampling_size), func=np.mean)
        dz = th.tensor(dz)
        # plt.imshow(dz, cmap="gray", vmin=-1, vmax=1)
        # plt.show()
        return dz

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.instances.iloc[idx, 0])
        image_raw = io.imread(img_name)[:240,:240,:]
        image = self.rgb2gray(image_raw)
        image = self.downsampling(image, self.downsampling_size)
        # from matplotlib import pyplot as plt
        # plt.imshow(image, cmap="gray", vmin=-1, vmax=1)
        # plt.show()

        labels = self.instances.iloc[idx, 1:]
        labels = th.tensor([labels], dtype=th.float32)
        sample = (image, labels)
        if self.transform:
            sample = self.transform(sample)
        return sample



def train(model, device, train_loader, optimizer, epoch, loss_function, f_log):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).squeeze(1)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)

        loss.backward()
        optimizer.step()
        f_log.write(f'Train Epoch: {epoch} [{(batch_idx+1) * len(data)}/{len(train_loader.dataset)} ({100. * (batch_idx+1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\n')



def test(model, device, test_loader, loss_function, f_log):
    model.eval()
    test_loss = 0
    correct = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    sig = nn.Sigmoid()
    with th.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).squeeze(1)
            output = model(data)
            test_loss += loss_function(output, target).item()  # sum up batch loss
            pred = (sig(output) > 0.5).float()
            correct += pred.eq(target.view_as(pred)).sum().item()
            true_positive  += th.logical_and(target.view_as(pred) == 1, pred == 1).sum().item()
            true_negative  += th.logical_and(target.view_as(pred) == 0, pred == 0).sum().item()
            false_positive += th.logical_and(target.view_as(pred) == 0, pred == 1).sum().item()
            false_negative += th.logical_and(target.view_as(pred) == 1, pred == 0).sum().item()
        assert true_positive+true_negative+false_positive+false_negative == 400
    test_loss /= len(test_loader.dataset)

    precision = (true_positive * 100)/(true_positive+false_positive) if true_positive+false_positive != 0 else -1
    recall = (true_positive * 100)/(true_positive+false_negative)
    f_log.write(f'Test set: Average loss: {test_loss:.4f}, \n\t\t' +
        f'Accuracy: {correct}/{len(test_loader.dataset) * 4} ({100. * correct / (len(test_loader.dataset) * 4):.0f}%)\n\t\t' +
        f'Precision: {true_positive}/{true_positive+false_positive} ({precision:.0f}%),\n\t\t' +
        f'Recall: {true_positive}/{true_positive+false_negative} ({recall:.0f}%), \n\t\t' +
        f'tp: {true_positive}, tn: {true_negative}, fp: {false_positive}, fn:{false_negative}\n')
    f_log.flush()

def calculate_sample_weights(dataset):
    keys = ["ghost(up)", "ghost(down)", "ghost(left)", "ghost(right)"]
    pos_weights = []
    for key in keys:
        ones = dataset.instances[key].value_counts()[1]
        zeros = dataset.instances[key].value_counts()[0]
        pos_weight = zeros/ones
        pos_weights.append(pos_weight)
    return th.tensor(pos_weights)

def main(csv_file, root_dir, model_folder, n_train, net_class, net_input_size, net_output_size, downsampling_size, epochs):
    use_cuda = False
    # epochs = 10000
    batch_size = 256
    device = th.device("cuda" if use_cuda else "cpu")
    th.manual_seed(0)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    dataset_train = Goal_Finding_Dataset(csv_file, root_dir, downsampling_size, train=True, n_train=n_train)
    dataset_test = Goal_Finding_Dataset(csv_file, root_dir, downsampling_size, n_train=n_train, n_test=100)

    train_loader = th.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    test_loader = th.utils.data.DataLoader(dataset_test, batch_size=batch_size)

    model = net_class(input_size=net_input_size, output_size=net_output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    pos_weight = calculate_sample_weights(dataset_train)
    loss_function = th.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    log_path = os.path.join(model_folder, f"observation_model_{n_train}_examples.log")
    f_log = open(log_path, "w")
    for epoch in range(1, epochs):
        train(model, device, train_loader, optimizer, epoch, loss_function, f_log)
        test(model, device, test_loader, th.nn.BCEWithLogitsLoss(reduction='sum'), f_log)
    model_path = os.path.join(model_folder, f"observation_model_{n_train}_examples.pt")
    th.save(model.state_dict(), model_path)



# if __name__ == '__main__':
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     folder = os.path.join(dir_path, "../data/gf3")
#     csv_file = os.path.join(folder, "labels.csv")
#     model_folder = os.path.join(folder)
#
#     # generate_random_images(csv_file, folder, n_images=1000)
#
#     # main(csv_file=csv_file, root_dir=folder, model_folder=model_folder, n_train=10)
#     # main(csv_file=csv_file, root_dir=folder, model_folder=model_folder, n_train=100)
#     main(csv_file=csv_file, root_dir=folder, model_folder=model_folder, n_train=1000)
