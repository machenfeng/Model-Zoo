import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class model(nn.Module):

    """ Short-Text-Pairs-Ranking: http://eecs.csuohio.edu/~sschung/CIS660/RankShortTextCNNACM2015.pdf """
    
    def __init__(self, args):
        super(model, self).__init__()

        vs = args.vocab_size
        ed = args.embed_dim
        
        ks = args.kernel_size
        oc = args.out_channels
        sl = args.seq_length
        
        ps = ks - 1  # padding size
        hs = 2 * oc + 1
        
        self.embed = nn.Embedding(vs, ed)
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=oc, kernel_size=(ks, ed), padding=(ps, 0)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=(sl - ks + 1 + ps * 2, 1)))
        
        self.m = nn.Parameter(torch.Tensor(oc, oc))
        self.m.data.uniform_(-0.1, 0.1)
        
        self.fc = nn.Sequential(nn.Linear(2 * oc + 1, hs), 
                                nn.BatchNorm1d(num_features=hs), 
                                nn.ReLU(), 
                                nn.Linear(hs, 2))     

    def forward(self, x1, x2):
        
        bs = x1.size(0)
        
        # (bs, sl) -> (bs, 1, sl, ed)
        x1 = self.embed(x1).unsqueeze(1)
        x2 = self.embed(x2).unsqueeze(1)
        
        # formula (1)
        # (bs, 1, sl, ed) -> (bs, oc)
        x1 = self.conv(x1).view(bs, -1)
        x2 = self.conv(x2).view(bs, -1)
        
        # formula (2)
        # sim.size() = (bs, 1)
        x1_ = x1.unsqueeze(1)
        x2_ = x2.unsqueeze(1).transpose(2, 1)
        m = self.m.expand(bs, -1, -1)
        sim = torch.bmm(x1_, m)
        sim = torch.bmm(sim, x2_).squeeze(2)
        
        # (bs, 2 * oc + 1)
        out = torch.cat([x1, sim, x2], 1)

        # (bs, 2 * oc + 1) -> (bs, hs)
        logits = self.fc(out)
        
        return logits
    
    @staticmethod
    def evaluate(model, dataloader, lossfunc):

        y_pred = []
        y_true = []
        total_loss = 0
        batch_count = len(dataloader)

        for (x1, x2, _, __, target) in dataloader:

            x1, x2 = x1.to(device), x2.to(device)
            target = target.view(-1).to(device)

            logits = model(x1, x2)
            predict = logits.max(1)[1]
            loss = lossfunc(logits, target)
            total_loss += loss.item()

            y_true += target.tolist()
            y_pred += predict.tolist()

        return y_true, y_pred, total_loss / batch_count
