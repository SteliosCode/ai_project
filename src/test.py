import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers
from keras import ops
from keras.models import Sequential
from keras.layers import Dense


#=========================================================#
#                       READ DATASET                        
#=========================================================#


# Column names (so we know what each number means)
COLUMN_NAMES = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","num"
]

# Read the file
df = pd.read_csv("processed.cleveland.data",
                  names=COLUMN_NAMES)

df.to_csv("full_dataset.csv", index=False)



#==========================================#
#               CLEAN DATA                 #
#==========================================#


data = pd.read_csv("processed.cleveland.data",
                   names=COLUMN_NAMES,
                   na_values=["?"])

clean_data = data.dropna()

print("""

=========================     
CLEAN DATA (FULL DATASET)
=========================      
       """ 
)
print(clean_data)



# Βεβαιώσου ότι το index είναι καθαρό
clean_data = clean_data.reset_index(drop=True)

sns.set_style("whitegrid")

# FacetGrid για το dataset σου
sns.FacetGrid(clean_data,
              hue="num",     # Χρώμα με βάση την κλάση (0–4)
              height=6
             ).map(
                 plt.scatter,
                 'age',       # Χ αξενας
                 'chol'       # Y άξονας
             ).add_legend()

plt.show()

# OPTIONAL: Save to files (uncomment if you want)
# X_train.to_csv("train_X.csv", index=False)
# y_train.to_csv("train_y.csv", index=False)
# X_val.to_csv("val_X.csv", index=False)
# y_val.to_csv("val_y.csv", index=False)
# X_test.to_csv("test_X.csv", index=False)
# y_test.to_csv("test_y.csv", index=False)