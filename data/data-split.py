from sklearn.model_selection import train_test_split¶

from os import listdir
from os.path import isfile, join

import pathlib
script_location = pathlib.Path(__file__).parent.resolve()
pos_path = str(script_location) + "\\raw\\positive"
neg_path = str(script_location) + "\\raw\\negative"

train_path = str(script_location) + "\\split\\train"
test_path = str(script_location) + "\\split\\test"
vali_path = str(script_location) + "\\split\\vali"

pos = [f for f in listdir(pos_path) if isfile(join(pos_path, f))]
neg = [f for f in listdir(neg_path) if isfile(join(neg_path, f))]

print(pos)
print(neg)

X_train, X_test, y_train, y_test = train_test_split(pos, neg, test_size=0.15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(pos, neg, test_size=0.17647058824, random_state=42) # to get a 70, 15, 15 split 0.15 / (1-0.15) = 0.17647058823