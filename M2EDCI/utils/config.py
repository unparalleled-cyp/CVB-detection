import argparse
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg

net_arg = add_argument_group("Network")
net_arg.add_argument('--char_emb_dim', type=int, default=50)
net_arg.add_argument('--lstm_hidden_dim', type=int, default=256)  # 256)
net_arg.add_argument('--lstm_layer', type=int, default=2)
net_arg.add_argument('--dropout_rate', type=float, default=0.3)
net_arg.add_argument('--gat_dropout_rate', type=float, default=0.3)
net_arg.add_argument('--gat_hidden_dim', type=int, default=256)  # 256)
net_arg.add_argument('--alpha', type=float, default=0.1)  
net_arg.add_argument('--gat_nheads', type=int, default=4) 
net_arg.add_argument('--gat_layer', type=int, default=2)

data_arg = add_argument_group("Data")
data_arg.add_argument("--fileholder", type=str, default=r"../dataset/data_bak/1.5-s125/") # 1.6-s1307-.97543/
data_arg.add_argument("--vocab_filename", type=str, default=r"../embeddings/vocab.txt")
data_arg.add_argument("--pretrained_char_emb_filename", type=str, default=r"../embeddings/weights.txt")
data_arg.add_argument("--param_stored_fileholder", type=str, default=r"saved_model")
# data_arg.add_argument("--dev_rate", type=float, default=0.3)
data_arg.add_argument("--char_alphabet_size", type=int, default=5789)  
data_arg.add_argument("--label_size", type=int, default=3)  

preprocess_arg = add_argument_group('Preprocess')
preprocess_arg.add_argument('--norm_char_emb', type=bool, default=False)  
preprocess_arg.add_argument('--norm_gaz_emb', type=bool, default=True)  
preprocess_arg.add_argument('--number_normalized', type=bool, default=True)  
preprocess_arg.add_argument('--max_sentence_length', type=int, default=250)

learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--batch_size', type=int, default=150)
learn_arg.add_argument('--max_epoch', type=int, default=50)
learn_arg.add_argument('--lr', type=float, default=0.001)
learn_arg.add_argument('--lr_decay', type=float, default=0.01)
learn_arg.add_argument("--optimizer", type=str, default="Adam", choices=['Adam', 'SGD'])
learn_arg.add_argument("--l2_penalty", type=float, default=0.00000005)  
learn_arg.add_argument("--log_path", type=str, default=r"log")

misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--mode', type=str, default="train", choices=["train", "val", "predict"])
misc_arg.add_argument('--shuffle', type=bool, default=True)
misc_arg.add_argument('--use_gpu', type=bool, default=True)
misc_arg.add_argument('--use_weights', type=bool, default=True)  
misc_arg.add_argument('--visible_gpu', type=str, default="0")
misc_arg.add_argument('--random_seed', type=int, default=1)
misc_arg.add_argument('--print_iter', type=int, default=1)  # 20
misc_arg.add_argument('--evaluate_train', type=int, default=5)  



def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed
