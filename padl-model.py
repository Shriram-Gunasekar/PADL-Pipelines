import padl
import re
import torch

# Define transformations
@padl.transform
def clean(text):
    return re.sub('[^A-Za-z\\ ]', ' ', text)

split_strip = padl.transform(lambda x: x.strip().split())

@padl.transform
class Dictionary:
    def __init__(self, d, default='<unk>'):
        self.d = d
        self.default = default

    def __call__(self, token):
        return self.d.get(token, self.default)

dictionary = Dictionary({'apple': 0, 'banana': 1, 'cat': 2, '<unk>': 3})
UNK = dictionary('<unk>')
MIN_LEN = 100

@padl.transform
def pad(x):
    return list(x) + [UNK for _ in range(MIN_LEN - len(x))]

@padl.transform
def truncate(x):
    return x[:MIN_LEN]

@padl.transform
def post_process_annotation(arg):
    return {False: 'BAD', True: 'GOOD'}[(arg > 0.5).item()]

lower_case = padl.same.lower()
to_tensor = padl.transform(lambda x: torch.tensor(x))

# Define the TextModel
@padl.transform
class TextModel(torch.nn.Module):
    def __init__(self, n_tokens, hidden_size, emb_dim):
        super().__init__()
        self.rnn = torch.nn.GRU(emb_dim, hidden_size=hidden_size, batch_first=True)
        self.embed = torch.nn.Embedding(n_tokens, emb_dim)
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, lens):
        hidden = self.rnn(self.embed(x))[0]
        last = torch.stack([hidden[i, lens[i] - 1, :] for i in range(hidden.shape[0])])
        return self.output(last)

# Create the TextModel layer
layer = TextModel(len(dictionary.d), 1024, 64)
