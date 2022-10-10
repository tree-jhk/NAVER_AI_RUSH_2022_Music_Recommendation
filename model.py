from torch import nn

# 모델 구성
class Net(nn.Module):
    def __init__(self, args):
        super(Net,self).__init__()
        num_items = args["num_items"] # args["num_items"] == 곡의 수
        latent_dim = args["latent_dim"] # args["latent_dim"] == 잠재 벡터 사이즈 크기 -> 1000으로 함.
        
        encoder_layer = nn.Linear(num_items, latent_dim, bias=True)
        decoder_layer = nn.Linear(latent_dim, num_items, bias=True)

        # 가중치 초기화
        nn.init.xavier_uniform_(encoder_layer.weight)
        nn.init.xavier_uniform_(decoder_layer.weight)
        
        self.encoder = nn.Sequential(
            encoder_layer,
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            decoder_layer,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded