import sys

from utility import evaluate, load_model, load_test_set_gtsdb, load_test_set_gtsrb

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python script.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    exec_mode = sys.argv[2]
    dataset = sys.argv[3]

    model = load_model(model_name, exec_mode)
    if dataset == 'gtsrb':
        x_test, y_test = load_test_set_gtsrb(dataset)
    elif dataset == 'gtsdb':
        x_test, y_test = load_test_set_gtsdb(dataset)
    else:
        print('Error')
        sys.exit()
    evaluate(model, x_test, y_test)
