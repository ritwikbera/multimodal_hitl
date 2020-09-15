import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models 

class AirSimBCModel(nn.Module):
    def __init__(self, **kwargs):
        super(AirSimBCModel, self).__init__()
        self.__dict__.update(kwargs) # use update instead of assignment when inheriting a class

        self.bn = nn.BatchNorm1d(num_features=self.input_dim, affine=True)

        self.fc1 = nn.Sequential(nn.Linear(self.input_dim, 64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(32, self.output_dim), nn.Tanh())

    def forward(self, x):
        x = self.bn(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class GazeModelUnit(nn.Module):
    def __init__(self, **kwargs):
        super(GazeModelUnit, self).__init__()
        self.__dict__.update(kwargs)

        self.bn = nn.BatchNorm1d(num_features=self.input_dim)

        self.fc1 = nn.Sequential(nn.Linear(self.input_dim, self.n_hidden), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU())
        self.fc_means = nn.Sequential(nn.Linear(self.n_hidden, 2), nn.ReLU())

    def forward(self, x):
        x = self.bn(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_means(x)

        return x

class JointModel(nn.Module):
    def __init__(self, **kwargs):
        super(JointModel, self).__init__()
        self.t2v_dim = 3

        # total number of unique commands accomodated by Embedding lookup table
        self.num_commands = 3
        self.__dict__.update(kwargs)

        self.input_dim = 484 + 512 + 16 # image vec size + depth vec size + kinematic vec size

        self.bn = nn.BatchNorm1d(num_features=self.input_dim)
        self.task2vec = nn.Embedding(num_embeddings=self.num_commands, embedding_dim=self.t2v_dim)

        # common backbone layers
        self.conv = FMP(stack_size=2)
        self.fc1 = nn.Sequential(nn.Linear(self.input_dim + self.t2v_dim, 128), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128,128), nn.ReLU())

        # action and gaze prediction heads
        self.ao_hidden = nn.Linear(128 + 2, 32)
        self.ao = nn.Linear(32, self.action_dim)

        self.go_hidden = nn.Sequential(nn.Linear(128,32), nn.ReLU())
        self.go = nn.Sequential(nn.Linear(32, 2), nn.ReLU())

    def forward(self, X):
        X, x_depth, x_kine, command = X['visual_features'], X['depth_features'], X['kinematic_features'], X['command']

        # Embedding layer is essentially a lookup table so requires LongTensor as input type
        if str(X.device) == 'cpu':
            command = torch.LongTensor(command)
        else:
            command = torch.cuda.LongTensor(command).to(X.device)

        X = self.conv(X)
        X = torch.cat((X, x_depth, x_kine), dim=-1)

        X = torch.cat((self.bn(X), self.task2vec(command)), dim=-1)

        embed = self.fc2(self.fc1(X))

        gaze_hidden = self.go_hidden(embed)
        gaze_pred = self.go(gaze_hidden)

        action = self.ao_hidden(torch.cat((embed, gaze_pred), dim=-1))
        action = self.ao(action)

        return (gaze_pred, action)

    def weights_init(self, m):
        # define different initializations for each layer
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.zero_()

class FMP(nn.Module):
    def __init__(self, **kwargs):
        super(FMP, self).__init__()
        self.__dict__.update(kwargs)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=128*self.stack_size, out_channels=32, kernel_size=3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3), nn.ReLU())

        self.conv_block = nn.Sequential(self.conv1, self.conv2, self.conv3)
        self.flatten = Flatten()

    def forward(self, x):
        return self.flatten(self.conv_block(x))

class MEM(nn.Module):
    def __init__(self, **kwargs):
        super(MEM, self).__init__()
        self.__dict__.update(kwargs)

        self.bn = nn.BatchNorm1d(num_features=self.input_dim) if self.use_batch_norm else nn.Identity()
        self.fc1 = nn.Linear(self.input_dim,1000)
        self.h1 = nn.Linear(1000,1000)
        self.fc2 = nn.Linear(1000,32)

    def forward(self, x):
        x = self.bn(x)
        return self.fc2(F.relu(self.h1(F.relu(self.fc1(x)))))
        

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.image_channels = image_channels
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(std.device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


def get_resnet(layers_from_output=1):
    resnet18 = models.resnet18(pretrained=True)
    modules = list(resnet18.children())[:-layers_from_output]
    resnet18 = nn.Sequential(*modules)
    for p in resnet18.parameters():
        p.requires_grad = False
    resnet18.eval()

    return resnet18.to('cuda' if torch.cuda.is_available() else 'cpu')

def get_depth_map_encoder(model_dict_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_model = VAE()
    if model_dict_path:
        depth_model.load_state_dict(torch.load(model_dict_path))

    depth_model.eval()

    return depth_model.to(device)

pt_feat_extractor = get_resnet(4)
depth_model = get_depth_map_encoder()
