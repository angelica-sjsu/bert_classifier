from DataLoader import EmbeddedSentences
from model import FC_Model
from embedder import sentence_embedder
import data_prep as dl
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

# dirs and file paths
train_embeddings = os.path.join('embeddings', 'train')
test_embeddings = os.path.join('embeddings', 'dev')
data_dir = 'bert_data'
train_data = os.path.join(data_dir, 'train.csv')
train_labels = os.path.join(data_dir, 'train_labels.csv')
test_data = os.path.join(data_dir, 'dev.csv')
test_labels = os.path.join(data_dir, 'dev_labels.csv')
trained_model = 'fc_trained.pt'

# model params
epochs = 25
batch_size = 64
learning_rate = 1e-4
input_size = 768
hidden_size = 1000
output_size = 1


def bert_classificator():

    if not os.path.exists(trained_model):
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
        model = FC_Model(input_size, hidden_size, output_size)
        # loss
        loss_fn = torch.nn.BCELoss()
        # loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        steps = len(train_data_loader)
        accuracy = 0
        for e in range(epochs):
            for i, (embeddings, labels) in enumerate(train_data_loader):
                # print(embeddings, labels)
                output = model(embeddings)
                loss = loss_fn(output.squeeze(), labels)
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
                    output = model(embeddings)
                    pred = output.data > 0.5
                    pred = pred.float()
                    total += labels.size(0)
                    correct += torch.sum(pred.squeeze() == labels)

                print(f'Total: {total}, correct:{correct}')
                if correct > accuracy:
                    # save model with the highest accuracy
                    torch.save(model.state_dict(), trained_model)

                accuracy = correct

    # reload model: the version with the highest number of correct classification tested against 10k reviews
    model = FC_Model(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(trained_model))
    # calling eval to set batch normalization before using the model against new data
    model.eval()

    #load new reviews and labels to test the model
    trial_embeddings = os.path.join('embeddings', 'test1')
    trial_data = pd.read_csv('my_reviews.csv')
    # call embedder to transform input data for the model
    if not os.path.exists(trial_embeddings):
        os.mkdir(trial_embeddings)
    print(f'[STATUS] -- creating trial embeddings ...')
    sentence_embedder(trial_data['reviews'], trial_embeddings, 'trial')
    embedded_trial = os.listdir(trial_embeddings)

    # feeding the new embeddings to the trained model.
    predictions = []
    for e in embedded_trial:
        path = os.path.join(trial_embeddings, e)
        sentence = np.load(path)
        predict = model(torch.from_numpy(sentence))
        if predict > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    # performance analysis
    # compare elements in prediction with the sentiments of the trial data
    # tp(true positive), tn
    correct = 0
    mis_class = 0
    length = len(predictions)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # let's look at precision and recall of the model: accuracy is not enough!!
    for p, sentiment in zip(predictions, trial_data['sentiment']):
        if p == sentiment:
            correct += 1
            if p == 0:
                tn += 1
            else:
                tp += 1
        else:
            mis_class += 1
            if p == 0:
                fn += 1
            else:
                fp += 1


    print(f'[ALERT] {correct} out of {length} reviews are classified correctly')
    print(f'[ALERT] {mis_class} out of {length} reviews are misclassified')
    # accuracy of positive prediction
    precision = tp/(tp + fp)
    # positive instances that are correctly detected by the classifier
    recall = tp/(tp + fn)
    # the classifier will only get a high F1 score if both recall and precision are high.
    f1 = 2/((1/precision) + (1/recall))
    print(f'[EVAL] precision = {precision}')
    print(f'[EVAL] recall = {recall}')
    print(f'[EVAL] F1 SCORE = {f1}')



    '''
    the accuracy of 83% was gathered by testing 10k reviews where 8.3k reviews were classified correctly.
    in this trial, we get a consistend 80% where 8 out of 10 reviews where reviewed correctly.
    However, recall is at 60% which is mid low values and consequently F1 score is mid-low. 
    F1 score is a better metrics than simply accuracy as it considers the ratios between true positive 
    and false negatives
    '''

if __name__ == '__main__':
    bert_classificator()
