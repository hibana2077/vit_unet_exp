import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGET_DIR = './OrganCMNIST'

# load all the results
results = {}
ext_data = {}
for file in os.listdir(TARGET_DIR):
    if file.endswith('.json'):
        with open(TARGET_DIR+'/'+file) as f:
            results[file.split('.')[0]] = json.load(f)

print(f"Total results: {len(results)}")

for key in results.keys():
    min_test_loss = np.min(results[key]['test_loss'])
    max_test_acc = np.max(results[key]['test_acc'])
    min_train_loss = np.min(results[key]['train_loss'])
    max_train_acc = np.max(results[key]['train_acc'])
    ext_data[key] = {
        'min_test_loss': min_test_loss,
        'max_test_acc': max_test_acc,
        'min_train_loss': min_train_loss,
        'max_train_acc': max_train_acc
    }

# convert to pandas dataframe
df = pd.DataFrame.from_dict(ext_data, orient='index')
df = df.sort_values(by=['max_test_acc'], ascending=False)
print(df)

# plot the results
plt.figure(figsize=(15,10))  # 調整圖表大小
plt.bar(df.index, df['max_test_acc'], label='Test Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title(f'Test Accuracy on {TARGET_DIR}')
plt.legend()
plt.xticks(rotation=45)  # 將 x 軸標籤旋轉角度調整為45度
plt.ylim(df['max_test_acc'].min()-1, df['max_test_acc'].max()+1)
plt.grid()
plt.tight_layout()  # 自動調整子圖參數以適應圖形區域
plt.savefig(TARGET_DIR+'/accuracy.png')

plt.figure(figsize=(12,10))
# plt.bar(df.index, df['min_train_loss'], label='Train Loss')
plt.bar(df.index, df['min_test_loss'], label='Test Loss')
plt.xlabel('Model')
plt.ylabel('Loss')
plt.title(f'Train and Test Loss on {TARGET_DIR}')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.savefig(TARGET_DIR+'/loss.png')