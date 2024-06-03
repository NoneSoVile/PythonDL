import pandas as pd
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--filename', default='accuracy.csv', type=str,
                    help='the file name of traning curve csv file')
args = parser.parse_args()
print("filename: ", args.filename)
# Load accuracy data from CSV file
data = pd.read_csv(args.filename)


# Plot training and validation accuracy
plt.figure(figsize=(10, 5), num=args.filename)
plt.plot(data['epoch'], data['train_acc1'], label='Train Acc@1')
plt.plot(data['epoch'], data['train_acc5'], label='Train Acc@5')
plt.plot(data['epoch'], data['val_acc1'], label='Val Acc@1')
plt.plot(data['epoch'], data['val_acc5'], label='Val Acc@5')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
