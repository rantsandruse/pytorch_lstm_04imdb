'''
This is the improved version of main.py
The main improvements are:
1. Now the input is a customizable csv, instead of hard coded in the text
2. Build a customizable training function.

'''
import torch.nn as nn
import torch.optim as optim

from pytorch_lstm_04imdb.model_lstm_classifier import LSTMClassifier
from pytorch_lstm_04imdb.preprocess import *
from pytorch_lstm_04imdb.preprocess_imdb import *
from pytorch_lstm_04imdb.train import train_val_test_split, train, eval, calc_metrics,  plot_loss_acc
from pytorch_lstm_04imdb.config import *


torch.manual_seed(1)

def main_experiment(hidden_size):

    # Data prep
    glove_dict = load_pretrained_embedding(GLOVE_EMBEDDING_PATH)
    # crawl_dict = load_pretrained_embedding(CRAWL_EMBEDDING_PATH)

    training_data = pd.read_csv("data/imdb_dataset/imdb_dataset.csv")
    training_data.sentiment = training_data.sentiment.apply(lambda x: 1 if x=='positive' else 0)
    training_data = training_data.sample(frac=1).reset_index(drop=True)

    reviews = data_to_reviews(training_data, "review", remove_stopwords = REMOVE_STOPWORD, add_stemmer=ADD_STEMMER)
    text_vocab = seqs_to_dictionary_v4(reviews)

    embedding_matrix = build_embedding_matrix(glove_dict, text_vocab, emb_dim=EMBEDDING_DIM)
    # For config_2.py, use both glove and fasttext
    # glove_embedding_matrix = build_embedding_matrix(glove_dict, text_vocab, emb_dim=EMBEDDING_DIM)
    # crawl_embedding_matrix = build_embedding_matrix(crawl_dict, text_vocab, emb_dim=EMBEDDING_DIM)
    # embedding_matrix = np.concatenate([crawl_embedding_matrix, glove_embedding_matrix], axis=-1)
    # be careful here.

    X_lens = np.array([len(x) if len(x) <= MAX_LEN else MAX_LEN for x in reviews])
    X = pad_sequences([seq_to_embedding(x, text_vocab) for x in reviews], maxlen = MAX_LEN, padding = "post", value = 0)
    y = np.array(training_data["sentiment"].tolist())

    train_dataset, val_dataset, test_dataset = train_val_test_split(X, X_lens, y)

    model = LSTMClassifier(embedding_matrix, hidden_size, OUTPUT_SIZE, batch_size = BATCH_SIZE)

    optimizer = optim.AdamW(model.parameters(), lr=LN_RATE)

    loss_fn = nn.BCEWithLogitsLoss()

    # Training
    train_loss, val_acc, train_acc, test_acc = train(model, train_dataset, val_dataset, test_dataset, loss_fn, optimizer, batch_size = BATCH_SIZE, n_epochs = 200, patience = -1)

    my_data = pd.DataFrame({"train_loss": train_loss,
                            "val_acc": val_acc,
                            "train_acc": train_acc,
                            "test_acc": test_acc })

    max_test_acc = max(test_acc)
    max_index = test_acc.index(max_test_acc)

    my_data.to_csv("./output3/output_hidden_"+ str(hidden_size)+ ".csv" )
    # Examine training results
    #plot_loss_acc(train_loss, val_loss, val_acc, output="./output/hidden_" + str(hidden_size) + "_")

    # Take a look at the precision/recall of the testing data.
    # pred_prob = predict(model, test_dataset, BATCH_SIZE)
    # y_test = test_dataset.tensors[1]
    # p_score, r_score, a_score, ra_score = calc_metrics(pred_prob, y_test)

    return train_acc[max_index], val_acc[max_index], max(test_acc)


val_acc_arr = []
test_acc_arr = []
train_acc_arr = []


# hidden_size_arr = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

hidden_size_arr = [2048]

for hidden_size in hidden_size_arr:
    train_acc, val_acc, test_acc = main_experiment(hidden_size)
    val_acc_arr.append(val_acc)
    test_acc_arr.append(test_acc)
    train_acc_arr.append(train_acc)

hidden_exp_csv = pd.DataFrame({"hidden_size_arr": hidden_size_arr,
                               "val_acc": val_acc_arr,
                               "test_acc":test_acc_arr,
                               "train_acc": train_acc_arr})

hidden_exp_csv.to_csv("./output3/hidden_size_all_2048.csv")


