import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F


device = 'cpu'


# ----------------- Encoder ----------------- #
class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        efficient_net = models.efficientnet_b0(pretrained=True)
        self.features = nn.Sequential(*list(efficient_net.children())[:-2])  
        self.conv = nn.Conv2d(1280, embed_size, kernel_size=1)  

    def forward(self, images):
        features = self.features(images)  
        features = self.conv(features)  
        B, C, H, W = features.size()
        features = features.flatten(start_dim=2).permute(0, 2, 1)  
        return features  # (B, H*W, embed_size)

# ----------------- Attention ----------------- #
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  
        alpha = self.softmax(att)  
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  
        return context, alpha

# ----------------- Decoder ----------------- #
class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, num_layers=2, dropout=0.5):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout) # Move dropout before LSTM
        self.attention = Attention(embed_size, hidden_size, attention_dim)
        self.lstm = nn.LSTM(embed_size + embed_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.init_h = nn.Linear(embed_size, hidden_size * num_layers)
        self.init_c = nn.Linear(embed_size, hidden_size * num_layers)
        self.fc.weight = self.embedding.weight  # Weight tying

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out).view(self.num_layers, mean_encoder_out.size(0), -1)
        c = self.init_c(mean_encoder_out).view(self.num_layers, mean_encoder_out.size(0), -1)
        return h, c

    def forward(self, encoder_out, captions):
        embeddings = self.dropout(self.embedding(captions))  # Apply dropout to embeddings
        batch_size = encoder_out.size(0)
        max_seq_len = captions.size(1)
        h, c = self.init_hidden_state(encoder_out)

        outputs = []
        for t in range(max_seq_len):
            context, alpha = self.attention(encoder_out, h[-1])
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1).unsqueeze(1)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(lstm_out.squeeze(1)))  # Apply dropout before the final layer
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs

# ----------------- Full Model ----------------- #
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, attention_dim=256, num_layers=1, dropout=0.5):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, attention_dim, num_layers, dropout)

    def forward(self, images, captions):
        encoder_out = self.encoder(images)  
        outputs = self.decoder(encoder_out, captions)  
        return outputs
    
# Load checkpoint function
def load_checkpoint(filename="checkpoint.pth"):
    checkpoint = torch.load(filename, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Extract hyperparameters (to recreate model correctly)
    hyperparams = checkpoint['hyperparameters']
    vocab_size = hyperparams['vocab_size']
    embed_size = hyperparams['embed_size']
    hidden_size = hyperparams['hidden_size']
    attention_dim = hyperparams['attention_dim']
    num_layers = hyperparams['num_layers']
    dropout = hyperparams['dropout']

    # Recreate the model with the same architecture
    model = ImageCaptioningModel(
        vocab_size=vocab_size, embed_size=embed_size, hidden_size=hidden_size,
        attention_dim=attention_dim, num_layers=num_layers, dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Load model weights and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']

    # Load word2idx and idx2word
    word2idx = checkpoint['word2idx']
    idx2word = checkpoint['idx2word']

    return model, optimizer, start_epoch, train_losses, val_losses, word2idx, idx2word

def generate_caption(model, image, word2idx, idx2word, device, max_length=20, beam_size=5):
    model.eval()
    start_token = word2idx["<start>"]
    end_token = word2idx["<end>"]
    pad_token = word2idx["<pad>"]

    with torch.no_grad():
        # Prepare image features
        image = image.unsqueeze(0).to(device)
        encoder_out = model.encoder(image)  # (1, num_pixels, embed_size)

        # Initialize hidden state for the decoder
        h, c = model.decoder.init_hidden_state(encoder_out)  # (num_layers, 1, hidden_size)

        # Initialize beam: (sequence, score, h, c)
        sequences = [[[start_token], 0.0, h.clone(), c.clone()]]
        seen_sequences = set()  # Track unique sequences to avoid duplicates

        for _ in range(max_length):
            all_candidates = []
            for seq, score, h_prev, c_prev in sequences:
                if seq[-1] == end_token:
                    all_candidates.append((seq, score, h_prev, c_prev))
                    continue

                # Prepare input
                input_tensor = torch.tensor([seq[-1]], device=device).unsqueeze(0)  # (1, 1)
                embeddings = model.decoder.embedding(input_tensor)  # (1, 1, embed_size)

                # Attention and LSTM
                context, _ = model.decoder.attention(encoder_out, h_prev[-1])  # Use last layer's hidden state
                lstm_input = torch.cat([embeddings, context.unsqueeze(1)], dim=2)
                lstm_out, (h_new, c_new) = model.decoder.lstm(lstm_input, (h_prev, c_prev))

                # Predict next word
                output = F.log_softmax(model.decoder.fc(lstm_out.squeeze(1)), dim=1)  # (1, vocab_size)

                # Get top-k candidates
                top_scores, top_indices = output.topk(beam_size, dim=1)
                for i in range(beam_size):
                    token = top_indices[0, i].item()
                    new_score = score + top_scores[0, i].item()
                    new_seq = seq + [token]

                    if tuple(new_seq) not in seen_sequences:  # Avoid duplicate sequences
                        all_candidates.append((new_seq, new_score, h_new.clone(), c_new.clone()))
                        seen_sequences.add(tuple(new_seq))

            # Sort candidates and keep top-k
            all_candidates = sorted(all_candidates, key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)  # Adjusted length normalization
            sequences = all_candidates[:beam_size]

            # Early stopping if all sequences end with <end>
            if all(seq[-1] == end_token for seq, _, _, _ in sequences):
                break

        # Select the best sequence
        best_seq = max(sequences, key=lambda x: x[1] / (len(x[0]) ** 0.7))[0]  # Length-normalized score
        caption = [idx2word[idx] for idx in best_seq if idx not in {pad_token, start_token, end_token}]
        return " ".join(caption)