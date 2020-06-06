from DataLoader import EmbeddedSentences
from model import FC_Model
from embedder import sentence_embedder
import data_prep as dl
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

train_embeddings = os.path.join('embeddings', 'train')
test_embeddings = os.path.join('embeddings', 'dev')
data_dir = 'bert_data'
train_data = os.path.join(data_dir, 'train.csv')
train_labels = os.path.join(data_dir, 'train_labels.csv')
test_data = os.path.join(data_dir, 'dev.csv')
test_labels = os.path.join(data_dir, 'dev_labels.csv')

# model params
epochs = 25
batch_size = 64
learning_rate = 1e-4
input_size = 768
hidden_size = 1000
output_size = 1


def bert_classificator():

    # inital stage: dataset split and preparation
    dataset = pd.read_csv('imdb_dataset.csv')
    # droppint the length of text since I will not use it.
    dataset = dataset.drop(['length of text'], axis=1)
    review_train, sentiment_train, review_test, sentiment_test = dl.splitter(dataset['review'], dataset['sentiment'])

    print(f'Training labels encoder ... ')
    enc_sentiment_train = dl.encoder(sentiment_train)
    print(f'Test labels encoder ...')
    enc_sentiment_test = dl.encoder(sentiment_test)

    train_bert = dl.bert_formatter(review_train, enc_sentiment_train)
    dev_bert = dl.bert_formatter(review_test, enc_sentiment_test)

    # saving train and dev dataframes into a tsv file, readable by BERT
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    train_bert.to_csv(train_data, index=True, header=True)
    dev_bert.to_csv(test_data, index=True, header=True)
    # end data prep

    # create embeddings to be fed to the model
    # load data
    training = pd.read_csv(train_data)
    dev = pd.read_csv(test_data)
    training = training.drop([f'Unnamed: 0'], axis=1)

    if not os.path.exists(train_embeddings):
        os.makedirs(train_embeddings)
        # create files containing embeddings: feature maps
        sentence_embedder(training['texts'], train_embeddings, 'train')

    if not os.path.exists(test_embeddings):
        os.makedirs(test_embeddings)
        sentence_embedder(dev['texts'], test_embeddings, 'dev')

    # create a separate csv for labels
    data = pd.read_csv(train_data)
    labels = pd.DataFrame(data['label'], columns=['label'])
    labels.to_csv(train_labels)

    dev_data = pd.read_csv(test_data)
    dev_labels = pd.DataFrame(dev_data['label'], columns=['label'])
    dev_labels.to_csv(test_labels)

    # read embeddings
    train_set = EmbeddedSentences(train_embeddings, train_labels, 'train')
    train_data_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_set = EmbeddedSentences(test_embeddings, test_labels, 'dev')
    test_data_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    # create model
    model = FC_Model(input_size, hidden_size, output_size).cuda()
    # loss
    loss_fn = torch.nn.BCELoss()
    #loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps = len(train_data_loader)
    for e in range(epochs):
        for i, (embeddings, labels) in enumerate(train_data_loader):
            #print(embeddings, labels)
            output = model(embeddings.cuda())
            loss = loss_fn(output.squeeze(), labels.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(e + 1, epochs, i + 1, steps, loss.item()))

        # test
        with torch.no_grad():
            correct = 0
            total = 0
            for embeddings, labels in test_data_loader:
                output = model(embeddings.cuda())
                pred = output.data > 0.5
                pred = pred.float()
                total += labels.size(0)
                correct += torch.sum(pred.squeeze() == labels.cuda())

            print(f'Total: {total}, correct:{correct}')

if __name__ == '__main__':
    bert_classificator()
