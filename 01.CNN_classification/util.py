import torch
import numpy as np


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        #  l2 norm (weight contraints): 3
        with torch.no_grad():
            for param in model.parameters():
                param.clamp_(min=-3, max=3)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def predict(TEXT, sentence, model, device, fixed_length=56):
    word2id = []

    for word in sentence.split():
        word2id.append(TEXT.vocab.stoi[word])

    word2id = word2id + [1] * (fixed_length - len(word2id))
    input_tensor = torch.LongTensor(word2id).to(device).unsqueeze(0)
    probability = np.squeeze(torch.sigmoid(model(input_tensor)).detach().numpy()[0], 0)
    predicted_label = 'Pos' if probability >= 0.5 else 'Neg'

    return probability, predicted_label


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    accuracy = correct.sum() / len(correct)

    return accuracy
