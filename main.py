import sys

from utility import evaluate, load_test_set_gtsrb, load_class_model

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python script.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    exec_mode = sys.argv[2]
    dataset = sys.argv[3]

    model = load_class_model(model_name, exec_mode)
    if dataset == 'gtsrb':
        x_test, y_test = load_test_set_gtsrb(dataset)
    else:
        print('Error')
        sys.exit()
    evaluate(model, x_test, y_test)
