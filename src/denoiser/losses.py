import torch
import torch.nn as nn
from speechbrain.processing.features import spectral_magnitude
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2

class STFTMagnitudeLoss(nn.Module):
    """
    Implementation of Spectral Magnitude Loss. 
    Computes the distance between two STFT representations.
    
    Arguments
    ---------
    distance_metric: str
        Which distance metric to use it (L1 or L2)
    compute_stft: object
        Function to compute the STFT representations.
    use_log: bool
        Wheter to use logarithmically-scaled loss
    """
    def __init__(
        self, 
        distance_metric:str="l2",
        compute_stft:object=None,
        use_log:bool=False
    ):
        super().__init__()
        self.distance_metric = distance_metric
        self.compute_stft = compute_stft
        self.use_log = use_log

        # Validate the passed parameters
        dist_types = ['l1', 'l2']
        
        if self.distance_metric not in dist_types:
            raise ValueError(
                f'distance_metric must be one of {dist_types}'
            )

        self.dist_metric = self.load_distance_metric()

    def forward(self, y_hat, y, lens=None):
        # Unfold predictions
        y_hat, y = y_hat.squeeze(-1), y.squeeze(-1)

        # Compute STFT representations
        rep_y_hat, rep_y = map(self.compute_features, [y_hat, y])
        loss = self.dist_metric(rep_y_hat, rep_y)

        if self.use_log:
            loss = 10*torch.log10(loss)

        return loss.mean()
    
    def load_distance_metric(self):
        fns = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss()
        }

        return fns[self.distance_metric]
    
    def compute_features(self, x):
        # Spectrogram
        feats = self.compute_stft(x)
        feats = spectral_magnitude(feats, power=0.5)

        return feats

class DeepFeatureLoss(nn.Module):
    """
    Implementation of Perceptual Losses for SE
    Based on the Phone-fortified Perceptual Loss
    Paper: https://arxiv.org/pdf/2010.15174v3.pdf
    Code: https://github.com/aleXiehta/PhoneFortifiedPerceptualLoss
    
    Arguments
    ---------
    PRETRAINED_MODEL_PATH: str
        Path where the weights of the model are stored
    alpha: float
        How much emphasis the loss gives on the latent-representation distance
    model_architecture: str
        Choose the model to extract representations from.
    """
    def __init__(self, 
        PRETRAINED_MODEL_PATH:str,
        alpha:float=10,
        compute_stft:object=None
    ):
        super().__init__()
        self.alpha = alpha
        self.PRETRAINED_MODEL_PATH = PRETRAINED_MODEL_PATH

        self.model = self.load_representation_model()
        self.dist_metric = nn.MSELoss()
        self.reconstruction_loss = STFTMagnitudeLoss(
            distance_metric="l1",
            compute_stft=compute_stft
        )

    def forward(self, y_hat, y, lens=None):
        # STFT magnitude loss
        l1_loss = self.reconstruction_loss(y_hat, y)

        # Unfold predictions
        y_hat, y = y_hat.squeeze(-1), y.squeeze(-1)

        # Compute latent representations
        with torch.no_grad():
            rep_y_hat, rep_y = map(self.model, [y_hat, y])

        dist = self.dist_metric(rep_y_hat, rep_y).mean()

        return l1_loss + self.alpha*dist
    

    def load_representation_model(self):
        model = HuggingFaceWav2Vec2(
            "facebook/wav2vec2-base-960h",
            save_path=self.PRETRAINED_MODEL_PATH
        )
        model.eval()

        return model#.to("cuda")

