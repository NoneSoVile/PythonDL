import csv
import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--world-size', default=9, type=int,
                    help='number of nodes for distributed training')
args = parser.parse_args()
print("world_size", args.world_size)
 # Initialize lists to store accuracy values
train_acc1_list = []
train_acc5_list = []
val_acc1_list = []
val_acc5_list = []
epochs = 100

for epoch in range(epochs):
    train_acc1_list.append(1.0 + epoch*0.7)
    train_acc5_list.append(2.0+ epoch*0.7)
    val_acc1_list.append(3.0+ epoch*0.7)
    val_acc5_list.append(4.0+ epoch*0.7)
 
# Save accuracy values to a CSV file
with open('accuracy.csv', 'w', newline='') as csvfile:
    fieldnames = ['epoch', 'train_acc1', 'train_acc5', 'val_acc1', 'val_acc5']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for epoch in range(epochs):
        writer.writerow({'epoch': epoch + 1,
                         'train_acc1': train_acc1_list[epoch],
                         'train_acc5': train_acc5_list[epoch],
                         'val_acc1': val_acc1_list[epoch],
                         'val_acc5': val_acc5_list[epoch]})