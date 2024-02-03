import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import os
import torch as th
import pandas as pd
import numpy as np
from skimage import io
from skimage.measure import block_reduce
from torch.utils.tensorboard import SummaryWriter
import time
import cv2


class Custom_Dataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_folder,
        image_dim,
        downsampling_size,
        train=False,
        transform=None,
        num_training_examples=400,
        num_test_examples=100,
    ):
        """

        :param csv_file: Path to the csv file with annotations.
        :param image_folder: Directory with all the images.
        :param image_dim:
        :param downsampling_size:
        :param train:
        :param transform: Optional transform to be applied
                on a sample.
        :param num_training_examples:
        :param num_test_examples:
        """

        self.n_train = num_training_examples
        self.n_test = num_test_examples
        self.image_dim = image_dim
        self.downsampling_size = downsampling_size
        if train:
            self.instances = pd.read_csv(csv_file)[: self.n_train]
        else:
            self.instances = pd.read_csv(csv_file)[
                self.n_train : self.n_train + self.n_test
            ]
        self.root_dir = image_folder
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.instances)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [-1, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            gray = gray / 128.0 - 1.0
        return th.tensor(gray, dtype=th.float32)

    @staticmethod
    def downsampling(x, downsampling_size):
        if downsampling_size is not None:
            dz = block_reduce(
                x.squeeze(),
                block_size=(downsampling_size, downsampling_size),
                func=np.mean,
            )
            dz = th.tensor(dz)
            return dz
        else:
            return x

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.instances.iloc[idx, 0])
        image_raw = io.imread(img_name)[: self.image_dim, : self.image_dim, :]
        # In case of grayScale images the len(img.shape) == 2
        if len(image_raw.shape) > 2 and image_raw.shape[2] == 4:
            # convert the image from RGBA2RGB
            image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGRA2BGR)
        image = self.rgb2gray(image_raw)
        image = self.downsampling(image, self.downsampling_size).unsqueeze(dim=0)

        labels = self.instances.iloc[idx, 1:]
        labels = th.tensor([labels], dtype=th.float32)
        sample = (image, labels)
        if self.transform:
            sample = self.transform(sample)
        return sample


num_iters_train = 0
num_iters_test1 = 0
num_iters_test2 = 0


def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    loss_function1,
    loss_function2,
    net_output_size,
    f_log,
    writer,
):
    start_time = time.time()
    model.train()

    global num_iters_train

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).squeeze(1)
        optimizer.zero_grad()
        output = model(data)
        labels_loss1 = output[:, :net_output_size]
        loss1 = loss_function1(labels_loss1, target[:, :net_output_size])

        if loss_function2 is not None:
            labels_loss2 = output[:, net_output_size:]
            loss2 = loss_function2(labels_loss2, target[:, net_output_size:])
            loss = loss1 + loss2
        else:
            loss = loss1

        loss.backward()
        optimizer.step()

        # log
        f_log.write(
            f"Train Epoch: {epoch} [{(batch_idx+1) * len(data)}/{len(train_loader.dataset)} "
          + f"({100. * (batch_idx+1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\n"
        )
        writer.add_scalar("Loss/train", loss.item(), num_iters_train)
        writer.add_scalar("Loss/train_loss1", loss1.item(), num_iters_train)
        if loss_function2 is not None:
            writer.add_scalar("Loss/train_loss2", loss2.item(), num_iters_train)
        num_iters_train += 1
    time_epoch = time.time() - start_time
    writer.add_scalar("Time/train_per_epoch", time_epoch, epoch)


def test(
    model,
    device,
    test_loader,
    epoch,
    loss_function1,
    loss_function2,
    net_output_size,
    f_log,
    writer,
    use_train_set=False,
):
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
            labels_loss1 = output[:, :net_output_size]
            labels_truth1 = target[:, :net_output_size]
            test_loss1 = loss_function1(
                labels_loss1, labels_truth1
            ).item()  # sum up batch loss
            if loss_function2 is not None:
                labels_loss2 = output[:, net_output_size:]
                test_loss2 = loss_function2(
                    labels_loss2, target[:, net_output_size:]
                ).item()
                test_loss = test_loss1 + test_loss2
            else:
                test_loss = test_loss1
            avg_test_loss += test_loss
            pred = (sig(labels_loss1) > 0.5).float()
            correct += (labels_truth1 == pred).sum().item()
            tp = th.logical_and(labels_truth1 == 1, pred == 1).sum().item()
            tn = th.logical_and(labels_truth1 == 0, pred == 0).sum().item()
            fp = th.logical_and(labels_truth1 == 0, pred == 1).sum().item()
            fn = th.logical_and(labels_truth1 == 1, pred == 0).sum().item()
            assert tp + tn + fp + fn == labels_loss1.numel()
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            # log
            if not use_train_set:
                writer.add_scalar("Loss/test", test_loss, num_iters_test1)
                writer.add_scalar("Loss/test_loss1", test_loss1, num_iters_test1)
                if loss_function2 is not None:
                    writer.add_scalar("Loss/test_loss2", test_loss2, num_iters_test1)
                num_iters_test1 += 1
            else:
                writer.add_scalar(
                    "Loss/test (use trainset)", test_loss, num_iters_test2
                )
                writer.add_scalar(
                    "Loss/test_loss1 (use trainset)", test_loss1, num_iters_test2
                )
                if loss_function2 is not None:
                    writer.add_scalar(
                        "Loss/test_loss2 (use trainset)", test_loss2, num_iters_test2
                    )
                num_iters_test2 += 1

    avg_test_loss /= len(test_loader.dataset)

    precision = (total_tp) / (total_tp + total_fp) if total_tp + total_fp != 0 else -1
    recall = (total_tp) / (total_tp + total_fn) if total_tp + total_fn != 0 else -1
    accuracy = correct / (len(test_loader.dataset) * net_output_size)

    # log
    time_epoch = time.time() - start_time
    if not use_train_set:
        f_log.write(
            f"Test set: Average loss: {test_loss:.4f}, \n\t\t"
            + f"Accuracy: {correct}/{len(test_loader.dataset) * target.size()[1]} "
            + f"({100. * correct / (len(test_loader.dataset) * target.size()[1]):.0f}%)\n\t\t"
            + f"Precision: {total_tp}/{total_tp+total_fp} ({100. * precision:.0f}%),\n\t\t"
            + f"Recall: {total_tp}/{total_tp+total_fn} ({100. * recall:.0f}%), \n\t\t"
            + f"tp: {total_tp}, tn: {total_tn}, fp: {total_fp}, fn:{total_fn}\n"
        )
        f_log.flush()
        writer.add_scalar("Test/precision", precision, epoch)
        writer.add_scalar("Test/recall", recall, epoch)
        writer.add_scalar("Test/accuracy", accuracy, epoch)
        writer.add_scalar("Time/test_per_epoch", time_epoch, epoch)
    else:
        writer.add_scalar("Train/precision", precision, epoch)
        writer.add_scalar("Train/recall", recall, epoch)
        writer.add_scalar("Train/accuracy", accuracy, epoch)
        writer.add_scalar("Time/test_per_epoch (use trainset)", time_epoch, epoch)


def main(
    csv_file,
    image_folder,
    model_folder,
    num_training_examples,
    net_class,
    net_input_size,
    net_output_size,
    image_dim,
    downsampling_size,
    epochs,
    keys,
    pretrain_w_extra_labels=True,
    num_test_examples=200,
):
    use_cuda = False
    batch_size = 8
    save_freq = 50
    device = th.device("cuda" if use_cuda else "cpu")
    th.manual_seed(0)

    dataset_train = Custom_Dataset(
        csv_file,
        image_folder,
        image_dim,
        downsampling_size,
        train=True,
        num_training_examples=num_training_examples,
    )
    dataset_test1 = Custom_Dataset(
        csv_file,
        image_folder,
        image_dim,
        downsampling_size,
        num_training_examples=num_training_examples,
        num_test_examples=num_test_examples,
    )
    dataset_test2 = Custom_Dataset(
        csv_file,
        image_folder,
        image_dim,
        downsampling_size,
        num_training_examples=0,
        num_test_examples=num_test_examples,
    )

    train_loader = th.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    test_loader1 = th.utils.data.DataLoader(dataset_test1, batch_size=batch_size)
    test_loader2 = th.utils.data.DataLoader(dataset_test2, batch_size=batch_size)

    def calculate_sample_weights(dataset, keys):
        pos_weights = []
        for key in keys[:net_output_size]:
            ones = dataset.instances[key].value_counts()[1]
            zeros = dataset.instances[key].value_counts()[0]
            total = ones + zeros
            n_classes = 2
            pos_weight = (1 / ones) * (total / n_classes)
            pos_weights.append(pos_weight)
            print(key, pos_weight)
        return th.tensor(pos_weights)

    print("CLASS WEIGHTS TRAIN:")
    pos_weight = calculate_sample_weights(dataset_train, keys)
    print("\nCLASS WEIGHTS TEST:")
    calculate_sample_weights(dataset_test1, keys)
    print("\nCLASS WEIGHTS TEST (USE TRAINSET):")
    calculate_sample_weights(dataset_test2, keys)


    model = net_class(input_size=net_input_size, output_size=net_output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function1 = th.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if pretrain_w_extra_labels:
        loss_function2 = th.nn.MSELoss()
    else:
        loss_function2 = None

    log_folder = os.path.join(
        model_folder, f"observation_model_{num_training_examples}_examples"
    )
    log_path = os.path.join(
        log_folder, f"observation_model_{num_training_examples}_examples.log"
    )

    writer = SummaryWriter(log_dir=log_folder)
    f_log = open(log_path, "w")
    for epoch in range(1, epochs):
        train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            loss_function1,
            loss_function2,
            net_output_size,
            f_log,
            writer,
        )
        test(
            model,
            device,
            test_loader1,
            epoch,
            loss_function1,
            loss_function2,
            net_output_size,
            f_log,
            writer,
        )
        test(
            model,
            device,
            test_loader2,
            epoch,
            loss_function1,
            loss_function2,
            net_output_size,
            f_log,
            writer,
            use_train_set=True,
        )
        if epoch % save_freq == 0:
            path = os.path.join(log_folder, f"observation_{epoch}_steps.pt")
            th.save(model.state_dict(), path)

    model_path = os.path.join(
        model_folder, f"observation_model_{num_training_examples}_examples.pt"
    )

    th.save(model.state_dict(), model_path)
