import torch
import torch.nn as nn


class model(nn.Module):
    
    def __init__(self, args):
        super(model, self).__init__()

        nc = args.num_classes
        hs = args.hidden_state
        vs = args.vocab_size
        
        self.sl = args.seq_length
        self.ed = args.embed_dim

        self.embed = nn.Embedding(vs, self.ed)

        self.maxpool = nn.MaxPool2d((self.sl, 1))
        self.fc = nn.Sequential(nn.Linear(self.ed * 2, hs),
                                nn.BatchNorm1d(num_features=hs),
                                nn.ReLU(),
                                nn.Linear(hs, nc))
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0, std=1)

        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0, std=1)
            m.bias.data.fill_(0.01)

        return

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.01)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
            
    def avgpool(self, x, mask4div):
        return torch.div(x.sum(1), mask4div)

    def forward(self, x, mask):

        bs = x.size(0)
        # get mask
        mask4div = mask.sum(1).unsqueeze(1).expand([bs, self.ed])

        mask4fill = abs(mask - 1).byte()
        mask4fill = mask4fill.unsqueeze(2).expand([bs, self.sl, self.ed])

        # (bs, sl) -> (bs, sl, ed)
        out = self.embed(x)
        out = out.masked_fill_(mask4fill, 0)

        # (bs, sl, ed) -> (bs, 1, ed)
        max = self.maxpool(out)
        avg = self.avgpool(out, mask4div)

        # (bs, 1, ed) -> (bs, ed * 2)
        max = max.squeeze(1)
        avg = avg.squeeze(1)
        out = torch.cat([max, avg], 1)

        # (bs, ed * 2) -> (bs, nc)
        logits = self.fc(out)

        return logits
