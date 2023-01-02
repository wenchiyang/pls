import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, is_vecenv_wrapped, VecMonitor
from torch.utils.data import Dataset
import os
import gym
import pacman_gym
from pacman_gym.envs.goal_finding import sample_layout
import csv
from pls.dpl_policies.pacman.util import get_ground_wall, get_agent_coord
from pls.dpl_policies.sokoban.util import get_ground_truth_of_box, get_ground_truth_of_corners
from pls.dpl_policies.sokoban.util import get_agent_coord as get_agent_coord_sokoban
from pls.dpl_policies.carracing.util import get_ground_truth_of_grass
import torch as th
import pandas as pd
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
import random
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import time
from pls.workflows.evaluate import load_model_and_env
import json
from time import sleep
import cv2



def load_policy_cr(folder, model_at_step):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)
    model, env = load_model_and_env(folder, config, model_at_step)
    return model, env



def generate_random_images_cr(csv_path, folder, n_images=10):
    policy_folder = os.path.join(os.path.dirname(__file__), "../../experiments_safety/carracing/map1/PPO/seed1")
    model, env = load_policy_cr(policy_folder, model_at_step=600000)

    deterministic=False
    render=True
    f_csv = open(csv_path, "w")
    writer = csv.writer(f_csv)
    writer.writerow(["image_name", "grass(in_front)","grass(on_the_left)", "grass(on_the_right)"])

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    observations = env.reset()
    current_lengths = 0
    n = 0
    while n < n_images:
        actions = model.predict(observations, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions[0])

        if render:
            for e in env.envs:
                e.env.render()

        if current_lengths % 50 == 0:
            img = env.envs[0].render(mode="state_pixels")
            gray_img = env.envs[0].render(mode="gray")
            gray_img = th.tensor(gray_img).unsqueeze(dim=0).unsqueeze(dim=1)

            path = os.path.join(folder, f"img{n:06}.png")

            plt.imsave(path, img)
            ground_truth_grass = get_ground_truth_of_grass(input=gray_img)
            row = [f"img{n:06}.png"] + ground_truth_grass.flatten().tolist()
            writer.writerow(row)
            f_csv.flush()
            n += 1

        current_lengths += 1



def sample_object_locations_sokoban(room_fixed, num_boxes):
    ### 0: wall, 1:floor, 2:target, 3: box_on_target, 4:box, 5:agent,
    # agent_on_target is expressed as (self.room_fixed == 2) & (self.room_state == 5)

    room = np.copy(room_fixed)

    # get locations of floor or target
    floor_coors = np.where(room==1)
    target_coors = np.where(room==2)

    listOfCoordinates= list(zip(floor_coors[0], floor_coors[1])) + list(zip(target_coors[0], target_coors[1]))
    sampled_locs = random.sample(listOfCoordinates, k=num_boxes+1)
    agent_loc = sampled_locs[0]
    box_locs = sampled_locs[1:]

    if room[agent_loc] in [1, 2]:
        room[agent_loc] = 5

    for box_loc in box_locs:
        if room[box_loc] in [1]:
            room[box_loc] = 4
        elif room[box_loc] in [2]:
            room[box_loc] = 3

    if np.count_nonzero(room == 4) + np.count_nonzero(room == 3) != 2:
        k=1

    return room

def generate_random_images_sokoban(csv_path, folder, n_images=10):
    WALL_COLOR = th.tensor([0], dtype=th.float32)
    FLOOR_COLOR = th.tensor([1 / 6], dtype=th.float32)
    BOX_TARGET_COLOR = th.tensor([2 / 6], dtype=th.float32)
    BOX_ON_TARGET_COLOR = th.tensor([3 / 6], dtype=th.float32)
    BOX_COLOR = th.tensor([4 / 6], dtype=th.float32)

    PLAYER_COLOR = th.tensor([5 / 6], dtype=th.float32)
    PLAYER_ON_TARGET_COLOR = th.tensor([1], dtype=th.float32)

    PLAYER_COLORS = th.tensor(([5 / 6], [1]))
    BOX_COLORS = th.tensor(([3 / 6], [4 / 6]))
    OBSTABLE_COLORS = th.tensor(([0], [3 / 6], [4 / 6]))

    f_csv = open(csv_path, "w")
    writer = csv.writer(f_csv)
    writer.writerow(["image_name", "box(up)", "box(down)", "box(left)", "box(right)", "corner(up)", "corner(down)", "corner(left)", "corner(right)", "agent_r", "agent_c"])

    cache_root = os.path.join(folder, "../../../")

    config = {
        "env_type": "Boxoban-Train-v0",
        "env_features": {
            "max_steps": 120,
            "penalty_for_step": -0.1,
            "penalty_box_off_target": -10,
            "reward_box_on_target": 10,
            "reward_finished": 10,
            "reward_last": 0,
            "dim_room": [10, 10],
            "num_boxes": 2,
            "render": False,
            "render_mode": "rgb_array",
            "action_size": 5,
            "difficulty": "medium",
            "split": "2box1map_large4",
            "cache_root": cache_root,
            "height": 160,
            "width": 160,
            "downsampling_size": 1
        }
    }
    random.seed(567)
    env_name = config["env_type"]
    env_args = config["env_features"]
    env = gym.make(env_name, **env_args)
    env.reset()

    num_boxes = config["env_features"]["num_boxes"]

    for n in range(n_images):
        env.select_room()
        room = sample_object_locations_sokoban(env.room_fixed, num_boxes)
        env.env.room_state = room
        # print(env.env.room_state)
        img = env.render(mode="rgb_array")

        path = os.path.join(folder, f"img{n:06}.jpeg")
        plt.imsave(path, img)

        tinyGrid = env.render("tiny_rgb_array")
        tinyGrid = th.tensor(tinyGrid).unsqueeze(0)
        ground_truth_box = get_ground_truth_of_box(
            input=tinyGrid, agent_colors=PLAYER_COLORS, box_colors=BOX_COLORS,
        )
        ground_truth_corner = get_ground_truth_of_corners(
            input=tinyGrid, agent_colors=PLAYER_COLORS, obsacle_colors=OBSTABLE_COLORS, floor_color=FLOOR_COLOR,
        )
        agent_r, agent_c = get_agent_coord_sokoban(tinyGrid, PLAYER_COLORS)
        ground_truth = th.cat((ground_truth_box, ground_truth_corner), 1).flatten().tolist() + [agent_r, agent_c]

        row = [f"img{n:06}.jpeg"] + ground_truth
        writer.writerow(row)
        f_csv.flush()
        if n % 10 == 0:
            print('Produce: {}/{} [({:.0f}%)]'.format(n, n_images, float(n)/n_images))

    f_csv.close()


def generate_random_images_pacman(csv_path, folder, n_images=10, ghost_distance=1, map_name="small"):
    WALL_COLOR = 0.25
    GHOST_COLOR = 0.5
    PACMAN_COLOR = 0.75
    FOOD_COLOR = 1
    f_csv = open(csv_path, "w")
    writer = csv.writer(f_csv)
    writer.writerow(["image_name", "ghost(up)", "ghost(down)", "ghost(left)", "ghost(right)", "agent_r", "agent_c"])
    config = {
        "env_type": "GoalFinding-v0",
        "env_features":{
            "layout": map_name,
            "reward_goal": 10,
            "reward_crash": 0,
            "reward_food": 0,
            "reward_time": -0.1,
            "render": False,
            "max_steps": 2000,
            "num_maps": 0,
            "seed": 567,
            'render_mode': "gray",
            "height": 482,
            "width": 482,
            "downsampling_size": 1,
            "background": "bg_small.jpg"
        }
    }
    env_name = config["env_type"]
    env_args = config["env_features"]
    env = gym.make(env_name, **env_args)
    env.env.gameDisplay = env.env.display
    env.env.rules.quiet = False

    num_ghosts = 30 if map_name == "small" else env.env.num_agents
    num_food = 30 if map_name == "small" else env.env.num_food
    for n in range(n_images):
        layout = sample_layout(
            env.layout.width,
            env.layout.height,
            num_ghosts,
            num_food,
            env.env.non_wall_positions,
            env.env.wall_positions,
            env.env.all_edges,
            check_valid=False
        )
        env.env.game = env.rules.newGame(
            layout,
            env.env.pacman,
            env.env.ghosts,
            env.env.gameDisplay,
            env.env.beQuiet,
            env.env.catchExceptions,
            env.env.symX,
            env.env.symY,
            env.env.background
        )
        env.game.start_game()
        env.env.render()


        img = env.game.compose_img("rgb")
        path = os.path.join(folder, f"img{n:06}.jpeg")
        plt.imsave(path, img)

        tinyGrid = env.game.compose_img("tinygrid")
        tinyGrid = th.tensor(tinyGrid).unsqueeze(0)
        ground_truth_ghost = get_ground_wall(tinyGrid, PACMAN_COLOR, GHOST_COLOR, ghost_distance)
        agent_r, agent_c = get_agent_coord(tinyGrid, PACMAN_COLOR)
        row = [f"img{n:06}.jpeg"] + ground_truth_ghost.flatten().tolist() + [agent_r, agent_c]
        writer.writerow(row)
        f_csv.flush()
        if n % 10 == 0:
            print('Produce: {}/{} [({:.0f}%)]'.format(n, n_images, float(n)/n_images))

    f_csv.close()

def generate_random_images_gf(csv_path, folder, n_images=10):
    WALL_COLOR = 0.25
    GHOST_COLOR = 0.5
    PACMAN_COLOR = 0.75
    FOOD_COLOR = 1
    f_csv = open(csv_path, "w")
    writer = csv.writer(f_csv)
    writer.writerow(["image_name", "ghost(up)", "ghost(down)", "ghost(left)", "ghost(right)", "agent_r", "agent_c"])
    config = {
        "env_type": "GoalFinding-v0",
        "env_features":{
            "layout": "small",
            "reward_goal": 10,
            "reward_crash": 0,
            "reward_food": 0,
            "reward_time": -0.1,
            "render": False,
            "max_steps": 2000,
            "num_maps": 0,
            "seed": 567,
            'render_mode': "gray",
            "height": 482,
            "width": 482,
            "downsampling_size": 1,
            "background": "bg_small.jpg"
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
            30, #env.env.num_agents,
            30, #env.env.num_food,
            env.env.non_wall_positions,
            env.env.wall_positions,
            env.env.all_edges,
            check_valid=False
        )
        env.env.game = env.rules.newGame(
            layout,
            env.env.pacman,
            env.env.ghosts,
            env.env.gameDisplay,
            env.env.beQuiet,
            env.env.catchExceptions,
            env.env.symX,
            env.env.symY,
            env.env.background
        )
        env.game.start_game()
        env.env.render()


        img = env.game.compose_img("rgb")
        path = os.path.join(folder, f"img{n:06}.jpeg")
        plt.imsave(path, img)

        tinyGrid = env.game.compose_img("tinygrid")
        tinyGrid = th.tensor(tinyGrid).unsqueeze(0)
        ground_truth_ghost = get_ground_wall(tinyGrid, PACMAN_COLOR, GHOST_COLOR)
        agent_r, agent_c = get_agent_coord(tinyGrid, PACMAN_COLOR)
        row = [f"img{n:06}.jpeg"] + ground_truth_ghost.flatten().tolist() + [agent_r, agent_c]
        writer.writerow(row)
        f_csv.flush()
        if n % 10 == 0:
            print('Produce: {}/{} [({:.0f}%)]'.format(n, n_images, float(n)/n_images))

    f_csv.close()

class Goal_Finding_Dataset(Dataset):
    """Goal Finding dataset."""
    def __init__(self, csv_file, root_dir, image_dim, downsampling_size,
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
        self.image_dim = image_dim
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
        if downsampling_size is not None:
            dz = block_reduce(x.squeeze(), block_size=(downsampling_size, downsampling_size), func=np.mean)
            dz = th.tensor(dz)
            # plt.imshow(dz, cmap="gray", vmin=-1, vmax=1)
            # plt.show()
            return dz
        else:
            return x

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.instances.iloc[idx, 0])
        image_raw = io.imread(img_name)[:self.image_dim,:self.image_dim,:]
        # In case of grayScale images the len(img.shape) == 2
        if len(image_raw.shape) > 2 and image_raw.shape[2] == 4:
            #convert the image from RGBA2RGB
            image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGRA2BGR)
        image = self.rgb2gray(image_raw)
        image = self.downsampling(image, self.downsampling_size).unsqueeze(dim=0)
        # from matplotlib import pyplot as plt
        # plt.imshow(image, cmap="gray", vmin=-1, vmax=1)
        # plt.show()

        labels = self.instances.iloc[idx, 1:]
        labels = th.tensor([labels], dtype=th.float32)
        sample = (image, labels)
        if self.transform:
            sample = self.transform(sample)
        return sample

num_iters_train = 0
num_iters_test1 = 0
num_iters_test2 = 0

def train(model, device, train_loader, optimizer, epoch, loss_function1, loss_function2, net_output_size, f_log, writer, use_agent_coord=True):
    start_time = time.time()
    model.train()
    global num_iters_train
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).squeeze(1)
        optimizer.zero_grad()
        output = model(data)
        fire_labels = output[:, :net_output_size]
        fire_loss = loss_function1(fire_labels, target[:, :net_output_size])

        if use_agent_coord:
            agent_coord = output[:, -2:]
            agent_coord_loss = loss_function2(agent_coord, target[:, -2:])
            loss = fire_loss + agent_coord_loss
        else:
            loss = fire_loss
        loss.backward()
        optimizer.step()

        # log
        f_log.write(f'Train Epoch: {epoch} [{(batch_idx+1) * len(data)}/{len(train_loader.dataset)} ({100. * (batch_idx+1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\n')
        writer.add_scalar('Loss/train', loss.item(), num_iters_train)
        writer.add_scalar('Loss/train_fire', fire_loss.item(), num_iters_train)
        if use_agent_coord:
            writer.add_scalar('Loss/train_agent_coord', agent_coord_loss.item(), num_iters_train)
        num_iters_train += 1
    time_epoch = (time.time() - start_time)
    writer.add_scalar('Time/train_per_epoch', time_epoch, epoch)



def test(model, device, test_loader, epoch, loss_function1, loss_function2, net_output_size, f_log, writer, use_train_set=False, use_agent_coord=True):
    start_time = time.time()
    model.eval()
    avg_test_loss = 0
    correct = 0
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    sig = nn.Sigmoid()
    global num_iters_test1, num_iters_test2
    with th.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device).squeeze(1)
            output = model(data)
            fire_labels = output[:, :net_output_size]
            fire_targets = target[:, :net_output_size]
            test_fire_loss = loss_function1(fire_labels, fire_targets).item()  # sum up batch loss
            if use_agent_coord:
                agent_coord = output[:, -2:]
                test_agent_coord_loss = loss_function2(agent_coord, target[:, -2:]).item()
                test_loss = test_fire_loss + test_agent_coord_loss
            else:
                test_loss = test_fire_loss
            avg_test_loss += test_loss
            pred = (sig(fire_labels) > 0.5).float()
            correct += (fire_targets == pred).sum().item()
            tp = th.logical_and(fire_targets == 1, pred == 1).sum().item()
            tn = th.logical_and(fire_targets == 0, pred == 0).sum().item()
            fp = th.logical_and(fire_targets == 0, pred == 1).sum().item()
            fn = th.logical_and(fire_targets == 1, pred == 0).sum().item()
            assert tp+tn+fp+fn == fire_labels.numel()
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            # log
            if not use_train_set:
                writer.add_scalar('Loss/test', test_loss, num_iters_test1)
                writer.add_scalar('Loss/test_fire_loss', test_fire_loss, num_iters_test1)
                if use_agent_coord:
                    writer.add_scalar('Loss/test_agent_coord_loss', test_agent_coord_loss, num_iters_test1)
                num_iters_test1 += 1
            else:
                writer.add_scalar('Loss/test (use trainset)', test_loss, num_iters_test2)
                writer.add_scalar('Loss/test_fire_loss (use trainset)', test_fire_loss, num_iters_test2)
                if use_agent_coord:
                    writer.add_scalar('Loss/test_agent_coord_loss (use trainset)', test_agent_coord_loss, num_iters_test2)
                num_iters_test2 += 1


    avg_test_loss /= len(test_loader.dataset)

    precision = (total_tp)/(total_tp+total_fp) if total_tp+total_fp != 0 else -1
    recall = (total_tp)/(total_tp+total_fn) if total_tp+total_fn != 0 else -1
    accuracy = correct / (len(test_loader.dataset) * net_output_size)



    # log
    time_epoch = (time.time() - start_time)
    if not use_train_set:
        f_log.write(f'Test set: Average loss: {test_loss:.4f}, \n\t\t' +
            f'Accuracy: {correct}/{len(test_loader.dataset) * target.size()[1]} ({100. * correct / (len(test_loader.dataset) * target.size()[1]):.0f}%)\n\t\t' +
            f'Precision: {total_tp}/{total_tp+total_fp} ({100. * precision:.0f}%),\n\t\t' +
            f'Recall: {total_tp}/{total_tp+total_fn} ({100. * recall:.0f}%), \n\t\t' +
            f'tp: {total_tp}, tn: {total_tn}, fp: {total_fp}, fn:{total_fn}\n')
        f_log.flush()
        writer.add_scalar('Test/precision', precision, epoch)
        writer.add_scalar('Test/recall', recall, epoch)
        writer.add_scalar('Test/accuracy', accuracy, epoch)
        writer.add_scalar('Time/test_per_epoch', time_epoch, epoch)
    else:
        writer.add_scalar('Train/precision', precision, epoch)
        writer.add_scalar('Train/recall', recall, epoch)
        writer.add_scalar('Train/accuracy', accuracy, epoch)
        writer.add_scalar('Time/test_per_epoch (use trainset)', time_epoch, epoch)

def calculate_sample_weights(dataset, keys):
    # pos_weights = []
    # for key in keys:
    #     ones = dataset.instances[key].value_counts()[1]
    #     zeros = dataset.instances[key].value_counts()[0]
    #     pos_weight = zeros/ones
    #     pos_weights.append(pos_weight)
    #     print(key, pos_weight)
    # return th.tensor(pos_weights)
    pos_weights = []
    for key in keys:
        ones = dataset.instances[key].value_counts()[1]
        zeros = dataset.instances[key].value_counts()[0]
        total = ones + zeros
        n_classes = 2
        pos_weight = (1 / ones) * (total / n_classes)
        pos_weights.append(pos_weight)
        print(key, pos_weight)
    return th.tensor(pos_weights)


def pre_train(csv_file, root_dir, model_folder, n_train, net_class, net_input_size, net_output_size, image_dim, downsampling_size, epochs, keys, use_agent_coord=True):
    use_cuda = False
    batch_size = 8
    save_freq = 50
    device = th.device("cuda" if use_cuda else "cpu")
    th.manual_seed(0)

    dataset_train = Goal_Finding_Dataset(csv_file, root_dir, image_dim, downsampling_size, train=True, n_train=n_train)
    dataset_test1 = Goal_Finding_Dataset(csv_file, root_dir, image_dim, downsampling_size, n_train=n_train, n_test=200)
    dataset_test2 = Goal_Finding_Dataset(csv_file, root_dir, image_dim, downsampling_size, n_train=0, n_test=200)

    train_loader = th.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    test_loader1 = th.utils.data.DataLoader(dataset_test1, batch_size=batch_size)
    test_loader2 = th.utils.data.DataLoader(dataset_test2, batch_size=batch_size)



    model = net_class(input_size=net_input_size, output_size=net_output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("CLASS WEIGHTS TRAIN:")
    pos_weight = calculate_sample_weights(dataset_train, keys)
    print("CLASS WEIGHTS TEST:")
    calculate_sample_weights(dataset_test1, keys)
    print("CLASS WEIGHTS TEST (USE TRAINSET):")
    calculate_sample_weights(dataset_test2, keys)


    loss_function1 = th.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_function2 = th.nn.MSELoss()

    if "cnn" in str(net_class):
        log_folder = os.path.join(model_folder, f"observation_model_{n_train}_examples_{downsampling_size}_cnn")
        log_path = os.path.join(log_folder, f"observation_model_{n_train}_examples_{downsampling_size}_cnn.log")

    else:
        log_folder = os.path.join(model_folder, f"observation_model_{n_train}_examples")
        log_path = os.path.join(log_folder, f"observation_model_{n_train}_examples.log")

    writer = SummaryWriter(log_dir=log_folder)
    f_log = open(log_path, "w")
    for epoch in range(1, epochs):
        train(model, device, train_loader, optimizer, epoch, loss_function1, loss_function2, net_output_size, f_log, writer, use_agent_coord=use_agent_coord)
        test(model, device, test_loader1, epoch, loss_function1, loss_function2, net_output_size, f_log, writer, use_agent_coord=use_agent_coord)
        test(model, device, test_loader2, epoch, loss_function1, loss_function2, net_output_size, f_log, writer, use_train_set=True)
        if epoch % save_freq == 0:
            path = os.path.join(log_folder, f"observation_{epoch}_steps.pt")
            th.save(model.state_dict(), path)
    if "cnn" in str(net_class):
        model_path = os.path.join(model_folder, f"observation_model_{n_train}_examples_{downsampling_size}_cnn.pt")
    else:
        model_path = os.path.join(model_folder, f"observation_model_{n_train}_examples.pt")
    th.save(model.state_dict(), model_path)


