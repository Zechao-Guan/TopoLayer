import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import ripser as Rips  # calculate PD
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system("wget %s  --no-check-certificate; unzip %s" % (www, zipfile))
        os.system("mv %s %s" % (zipfile[: -4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition):
    """
    :param partition: train / test
    :return all_data, all_label:
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    all_data = []
    all_label = []

    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, "r")
        data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64")
        f.close()
        all_data.append(data)
        all_label.append(label)

    # 合并数据
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translate_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype("float32")
    return translate_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class PDDataStructure():
    def __init__(self, pointcloud, maxdim=2, thresh=0.5, coeff=2, distance_matrix=False, do_cocycles=False,
                 metric="euclidean", n_perm=128):
        ripser = Rips.ripser(pointcloud, maxdim=maxdim, thresh=thresh, coeff=coeff,
                             distance_matrix=distance_matrix, do_cocycles=do_cocycles,
                             metric=metric, n_perm=n_perm)
        # pd_data
        ripser["dgms"][1][np.isinf(ripser["dgms"][1])] = 2
        ripser["dgms"][1] = np.concatenate((ripser["dgms"][1], [[0, 0]]), axis=0)
        ripser["dgms"][2][np.isinf(ripser["dgms"][2])] = 2
        ripser["dgms"][2] = np.concatenate((ripser["dgms"][2], [[0, 0]]), axis=0)

        self.h1_pd_data = torch.tensor(ripser["dgms"][1], dtype=torch.float32)
        self.h2_pd_data = torch.tensor(ripser["dgms"][2], dtype=torch.float32)
        self.pd_data = (self.h1_pd_data, self.h2_pd_data)


def save_pd():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PD_DIR = os.path.join(DATA_DIR, "pd")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(PD_DIR):
        os.makedirs(PD_DIR)

    # train
    train_data, train_label = load_data(partition="train")

    for i, data in enumerate(tqdm(train_data)):
        h1_pd_data, h2_pd_data = PDDataStructure(data).pd_data
        save_h1_path = os.path.join(PD_DIR, f"h1_pd_data_train_{i}.h5")
        save_h2_path = os.path.join(PD_DIR, f"h2_pd_data_train_{i}.h5")
        with h5py.File(save_h1_path, "w") as hf:
            hf.create_dataset("h1_pd_data", data=h1_pd_data)
        with h5py.File(save_h2_path, "w") as hf:
            hf.create_dataset("h2_pd_data", data=h2_pd_data)

    # test
    test_data, test_label = load_data(partition="test")

    for i, data in enumerate(tqdm(test_data)):
        h1_pd_data, h2_pd_data = PDDataStructure(data).pd_data
        save_h1_path = os.path.join(PD_DIR, f"h1_pd_data_test_{i}.h5")
        save_h2_path = os.path.join(PD_DIR, f"h2_pd_data_test_{i}.h5")
        with h5py.File(save_h1_path, "w") as hf:
            hf.create_dataset("h1_pd_data", data=h1_pd_data)
        with h5py.File(save_h2_path, "w") as hf:
            hf.create_dataset("h2_pd_data", data=h2_pd_data)


def load_pd_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    h1_pd_data = []
    h2_pd_data = []

    if not os.path.exists(os.path.join(DATA_DIR, 'pd')):
        save_pd()
    # save_pd()

    for h5_name in glob.glob(os.path.join(DATA_DIR, 'pd', 'h1_pd_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, "r")
        data = f["h1_pd_data"][:]
        f.close()
        h1_pd_data.append(data)

    for h5_name in glob.glob(os.path.join(DATA_DIR, 'pd', 'h2_pd_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, "r")
        data = f["h2_pd_data"][:]
        f.close()
        h2_pd_data.append(data)

    return list(zip(h1_pd_data, h2_pd_data))


class ModelNet40(Dataset):
    def __init__(self, num_points, partition="train", opt="offline"):
        """
        :param opt: the type of data, "realtime" represents data not with PD, when the TopoPointNet calculates the real-time PD of data
                                      "delay" represents data with PD, where the input data of TopoPintNet include PD, but it still realtime
                                      "offline" represents data with PD, which PD is calculated for each data in advance and stored
        """
        self.opt = opt
        if opt == "offline":
            self.data, self.label = load_data(partition)
            B, N, D = self.data.shape
            noise = np.random.uniform(low=-0.1, high=0.1, size=[B, N, D]).astype("float32") # add noise
            self.data = self.data + noise
            self.pd_data = load_pd_data(partition)
            self.num_points = num_points
            self.partition = partition
        else:
            raise Exception("opt should be offline")

    def __getitem__(self, item):
        if self.opt == "offline":
            pointcloud = self.data[item][:self.num_points]
            pd_pointcloud = self.pd_data[item]
            label = self.label[item]
            if self.partition == "train":
                # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
                pointcloud = translate_pointcloud(pointcloud)
                np.random.shuffle(pointcloud)
            return pointcloud, pd_pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class Collatet():
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, batch):
        if self.opt == "offline":
            processed_pointcloud = torch.utils.data.dataloader.default_collate([item[0] for item in batch])
            processed_pd_data = [
                (torch.tensor(item[1][0], dtype=torch.float32), torch.tensor(item[1][0], dtype=torch.float32)) for item
                in batch]
            processed_label = torch.utils.data.dataloader.default_collate([item[2] for item in batch])
            return processed_pointcloud, processed_pd_data, processed_label

        else:
            raise Exception("opt should be offline")





