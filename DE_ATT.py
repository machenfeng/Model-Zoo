import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model(nn.Module):
    
    """ https://arxiv.org/pdf/1606.01933.pdf """

    def __init__(self, args):
        super(Model, self).__init__()
        
        ed = args.embed_dim  # encoder dim
        self.nc = args.num_classes
        self.hs = args.hidden_state  # 200 according to thesis

        self.al = args.a_length  # less than embed dim
        self.bl = args.b_length

        # vanilla version: embedding only
        self.embed = nn.Embedding.from_pretrained(args.pretrained_weights, freeze=True)
        self.projection = nn.Linear(ed, self.hs, bias=False)  # projection from page 4, Embedding
        self.f = self.mlp(self.hs, self.hs)  # denotes F(·) in formula (1)
        self.g = self.mlp(self.hs * 2, self.hs)  # denotes G(·) in formula (3)
        self.h = self.mlp(self.hs * 2, self.hs)  # denotes H(·) in formula (5)

        self.fc = nn.Linear(self.hs, self.nc)
    
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0, std=0.01)
                if m.bias is True:
                    m.bias.data.normal_(mean=0, std=0.01)

    def mlp(self, input_size, hidden_size):
        fc = nn.Sequential(nn.Dropout(0.2),
                 nn.Linear(input_size, hidden_size),
                 nn.ReLU(),
                 nn.Dropout(0.2),
                 nn.Linear(hidden_size, hidden_size),
                 nn.ReLU())
        return fc

    def forward(self, a, b, a_mask, b_mask):
        
        bs = a.size(0)
        # encode
        a = self.embed(a)
        b = self.embed(b)
        
        a = self.projection(a)  # denotes a_ in thesis
        b = self.projection(b)  # denotes b_ in thesis
        
        # formula (1)
        f_a = self.f(a)  # (bs, al, hs)
        f_b = self.f(b)  # (bs, bl, hs)
        e = torch.bmm(f_a, f_b.transpose(1, 2))  # (bs, al, bl), e_ij denotes unnormalized attention weight of (ai, bj)
        
        # formula (2)
        b_prob = nn.Softmax(dim=2)(e)  # (bs, al, bl), probability distribution of b given a
        a_prob = nn.Softmax(dim=1)(e).transpose(1, 2)  # (bs, bl, al), probability distribution of a given b
        
        beta = torch.bmm(b_prob, b)  # (bs, al, hs), b given a
        alpha = torch.bmm(a_prob, a)  # (bs, bl, hs), a given b

        # formula (3)
        a_cat_beta = torch.cat((a, beta), dim=2)  # (bs, al, hs * 2)
        b_cat_alpha = torch.cat((b, alpha), dim=2)  # bs, bl, hs * 2)
        
        v1 = self.g(a_cat_beta)  # (bs, al, hs)
        v2 = self.g(b_cat_alpha)  # (bs, bl, hs)
        
        # formula (4)
        v1 = torch.sum(v1, 1)  # (bs, hs)
        v2 = torch.sum(v2, 1)  # (bs, hs)
        
        # formula (5)
        out = torch.cat([v1, v2], 1)
        y_hat = self.h(out)  # (bs, hs * 2)
        logits = self.fc(y_hat)

        return logits

    @staticmethod
    def evaluate(model, dataloader, lossfunc):

        y_pred = []
        y_true = []
        total_loss = 0
        batch_count = len(dataloader)

        for (x1, x2, x1_mask, x2_mask, label) in dataloader:

            x1 = x1.to(device)
            x1_mask = x1_mask.to(device)
            x2 = x2.to(device)
            x2_mask = x2_mask.to(device)
            label = label.view(-1).to(device)

            logits = model(x1, x2, x1_mask, x2_mask)
            loss = lossfunc(logits, label)
            total_loss += loss.item()

            predict = logits.max(1)[1]
            y_true += label.tolist()
            y_pred += predict.tolist()

        return y_true, y_pred, total_loss / batch_count