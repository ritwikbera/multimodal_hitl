import os
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class AirSimDataset(Dataset):
    """AirSim dataset containing language, image, joystick, and gaze data."""

    def __init__(self, csv_file, root_dir, use_gaze=False, transform=None, resnet_model=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # dataset parameters
        self.log = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.resnet_model = resnet_model
        self.use_gaze = use_gaze

    def __len__(self):
        return len(self.log)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load image and fix color (RGBA to RGB)
        img_name = os.path.join(self.root_dir, f'rgb_{idx}.png')
        image = np.array(Image.open(img_name).convert('RGB'))

        # normalize it and convert to tensor
        if self.transform:
            image = self.transform(image).unsqueeze(0)

        # extract resnet features
        resnet_features = self.resnet_model(image).view(1,-1)

        # parse states and actions
        states = torch.Tensor([self.log.iloc[idx, 9:-7]])
        actions = torch.Tensor([self.log.iloc[idx, -7:-3]])

        # TODO: standardize states and actions

        # setup inputs and labels
        inputs = torch.cat((resnet_features, states), dim=1)
        labels = actions

        # check if using gaze as a input features
        if not self.use_gaze:
            # not using gaze, leave inputs the way it is
            return inputs, labels
        else:
            # using gaze, read and append it to inputs and return
            gaze_data = torch.Tensor([self.log.iloc[idx, -3:-1]])
            inputs = torch.cat((inputs, gaze_data), dim=1)

            return inputs, labels


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.squeeze().cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.squeeze().cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.squeeze().cpu().data.numpy()

        # print(np.unique(labels, return_counts=True))

        triplets = []
        for label in set(labels):
            label_mask = (label == labels) # elements within numpy should be hashable eg. int

            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


if __name__=='__main__':
    print('test run')