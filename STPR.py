import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model(nn.Module):
    """
    Short-Text-Pairs-Ranking
    http://eecs.csuohio.edu/~sschung/CIS660/RankShortTextCNNACM2015.pdf
    """
    
    def __init__(self, args):
        super(Model, self).__init__()

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

    
    def forward(self, q, a):
        
        bs = q.size(0)
        
        # (bs, sl) -> (bs, 1, sl, ed)
        q = self.embed(q).unsqueeze(1)
        a = self.embed(a).unsqueeze(1)
        
        # formula (1)
        # (bs, 1, sl, ed) -> (bs, oc)
        q = self.conv(q).view(bs, -1)
        a = self.conv(a).view(bs, -1)
        
        # formula (2)
        # sim.size() = (bs, 1)
        q_ = q.unsqueeze(1)
        a_ = a.unsqueeze(1).transpose(2, 1)
        m = self.m.expand(bs, -1, -1)
        sim = torch.bmm(q_, m)
        sim = torch.bmm(sim, a_).squeeze(2)
        
        # (bs, 2 * oc + 1)
        out = torch.cat([q, sim, a], 1)

        # (bs, 2 * oc + 1) -> (bs, hs)
        logits = self.fc(out)
        
        return logits


def eval(model, dataloader, lossfunc):

    y_pred = []
    y_true = []
    total_loss = 0
    batch_count = len(dataloader)
    
    for (q, a, _, __, target) in dataloader:

        q = q.to(device)
        a = a.to(device)
        target = target.view(-1).to(device)
        
        logits = model(q, a)
        predict = logits.max(1)[1]
        loss = lossfunc(logits, target)
        total_loss += loss.item()

        y_true += target.tolist()
        y_pred += predict.tolist()
    
    return y_true, y_pred, total_loss / batch_count
