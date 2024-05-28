import pandas as pd
import matplotlib.pyplot as plt

# Load accuracy data from CSV file
data = pd.read_csv('accuracy.csv')

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(data['epoch'], data['train_acc1'], label='Train Acc@1')
plt.plot(data['epoch'], data['val_acc1'], label='Val Acc@1')
plt.plot(data['epoch'], data['train_acc5'], label='Train Acc@5')
plt.plot(data['epoch'], data['val_acc5'], label='Val Acc@5')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
