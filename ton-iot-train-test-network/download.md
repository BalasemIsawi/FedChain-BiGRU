#### Downloading ToN-IoT Network #### 
**1. From Kaggle** 

```python
import pandas as pd
import numpy as np

file_path = '/kaggle/input/datasets/arnobbhowmik/ton-iot-network-dataset/train_test_network.csv'

df = pd.read_csv(file_path, nrows=500000)

print(f"Data Shape: {df.shape}")
df.head()
```
 
**2. UNSW Canberra at ADFA:** 

URL: [Go to page](https://research.unsw.edu.au/projects/toniot-datasets)