import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import TransformedDistribution
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal 

class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()
        
class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        assert len(triplets) > 0

        if embeddings.is_cuda:
            triplets = triplets.cuda()
    
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)

        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean()

class GazeKLD(nn.Module):
    '''
    KL Divergence Loss between two gaze heatmaps
    Takes in batch of true distributions (computed from consecutive gazepoint sequences) 
    and predicted distributions (computed from prediction mean and variances)
    '''

    def __init__(self):
        super(GazeKLD, self).__init__()
        # solves numerical instability when variance equals zero (true or pred)
        self.eps = 1e-6

    def forward(self, pred, true):
        # segment mean and diagonal variance values
        pred_means = pred[:,:2]
        pred_vars = pred[:,2:]

        true_means = true.mean(dim=1)
        true_vars = true.var(dim=1)

        def kld_single(i):
            true_dist = MultivariateNormal(true_means[i], torch.diag(true_vars[i]+self.eps))
            pred_dist = MultivariateNormal(pred_means[i], torch.diag(pred_vars[i]+self.eps))
            return kl_divergence(true_dist, pred_dist)

        return torch.mean(torch.stack([kld_single(i) for i in range(true_means.size(0))]))


class GazeKLDUnit(nn.Module):
    '''
    KL Divergence Loss between two gaze heatmaps
    Takes in batch of true distributions (computed from consecutive gazepoint sequences) 
    and predicted distributions (computed from prediction mean and variances)
    '''

    def __init__(self):
        super(GazeKLDUnit, self).__init__()
        # solves numerical instability when variance equals zero (true or pred)
        self.eps = 1e-6

    def forward(self, pred, true):
        # segment mean and diagonal variance values
        pred_means = pred[:,:2]
        true_means = true.mean(dim=1)
        self.unit_cov = torch.diag(torch.Tensor([1., 1.]))

        # send cov to same device as mean
        self.unit_cov = self.unit_cov.to(true_means.device)

        def kld_single(i):
            true_dist = MultivariateNormal(true_means[i], self.unit_cov)
            pred_dist = MultivariateNormal(pred_means[i], self.unit_cov)
            return kl_divergence(true_dist, pred_dist)

        return torch.mean(torch.stack([kld_single(i) for i in range(true_means.size(0))]))

class JointLoss(nn.Module):
    '''
    Weighted combined loss
    '''
    def __init__(self, weights=[0.5,0.5]):
        super(JointLoss, self).__init__()
        self.weights = weights
        # self.loss_fn_1 = GazeKLDUnit()
        # use below line for instantaneous gaze prediction
        self.loss_fn_1 = nn.MSELoss()
        self.loss_fn_2 = nn.MSELoss()

    def forward(self, pred, true):
        gaze_pred, action_pred = pred
        gaze_true, action_true = true 

        loss1 = self.loss_fn_1(gaze_pred, gaze_true)
        loss2 = self.loss_fn_2(action_pred, action_true)
        
        return self.weights[0]*loss1 + self.weights[1]*loss2


class VAE_loss(nn.Module):
    def __init__(self, mode='mse'):
        super(VAE_loss, self).__init__()
        self.mode = mode 

    def forward(self, pred, true):
        recon_x, mu, logvar = pred 
        x = true
        
        if self.mode == 'mse':
            BCE = F.mse_loss(recon_x, x, reduction='sum')
        else:
            BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD