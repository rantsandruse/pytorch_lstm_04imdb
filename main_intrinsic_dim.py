'''
This is the improved version of main.py
The main improvements are:
1. Now the input is a customizable csv, instead of hard coded in the text
2. Build a customizable training function.

'''
import os
import torch.nn as nn
import torch.optim as optim

from pytorch_lstm_04imdb.model_lstm_classifier import LSTMClassifier
from pytorch_lstm_04imdb.preprocess import *
from pytorch_lstm_04imdb.preprocess_imdb import *
from pytorch_lstm_04imdb.train import train_val_test_split, train, predict, calc_metrics, plot_loss_acc
from pytorch_lstm_04imdb.config import *

from pytorch_lstm_04imdb.intrinsic_dim.fastfood import FastfoodWrap
# from pytorch_lstm_04imdb.intrinsic_dim.dense import DenseWrap


torch.manual_seed(1)


def run_experiment(dim):
    HIDDEN_DIM=2
    # Data prep
    glove_dict = load_pretrained_embedding(GLOVE_EMBEDDING_PATH)

    training_data = pd.read_csv("data/imdb_dataset/imdb_dataset.csv")
    training_data.sentiment = training_data.sentiment.apply(lambda x: 1 if x=='positive' else 0)
    training_data = training_data.sample(frac=1).reset_index(drop=True)

    reviews = data_to_reviews(training_data, "review", remove_stopwords = REMOVE_STOPWORD, add_stemmer=ADD_STEMMER)
    text_vocab = seqs_to_dictionary_v4(reviews)

    embedding_matrix = build_embedding_matrix(glove_dict, text_vocab, emb_dim=EMBEDDING_DIM)

    X_lens = np.array([len(x) if len(x) <= MAX_LEN else MAX_LEN for x in reviews])
    X = pad_sequences([seq_to_embedding(x, text_vocab) for x in reviews], maxlen = MAX_LEN, padding = "post", value = 0)
    y = np.array(training_data["sentiment"].tolist())

    train_dataset, val_dataset, test_dataset = train_val_test_split(X, X_lens, y)

    model = LSTMClassifier(embedding_matrix, HIDDEN_DIM, OUTPUT_SIZE, batch_size = BATCH_SIZE).cuda()

    device=torch.device("cuda")
    model = FastfoodWrap(model, intrinsic_dimension=dim, device=device)

    # Experiment:
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training
    train_loss, val_loss, val_acc = train(model, train_dataset, val_dataset, loss_fn, optimizer, batch_size = BATCH_SIZE, n_epochs = 500, patience = -1)

    # Examine training results
    newdir = './output_hiddendim_2/dim_' + str(dim) + "/"
    if not os.path.exists(newdir):
        os.mkdir(newdir)
    plot_loss_acc(train_loss, val_loss, val_acc, output=newdir)

    # Take a look at the precision/recall of the testing data.
    # pred_prob = predict(model, test_dataset, BATCH_SIZE)
    # y_test = test_dataset.tensors[1]
    # calc_metrics(pred_prob, y_test)

# Run experiment
# for dim in [50, 100, 200, 300, 400, 500]:
for dim in [500, 600]:
    run_experiment(dim)


