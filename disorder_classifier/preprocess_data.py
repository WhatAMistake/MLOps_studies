import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.utils
from sklearn.model_selection import train_test_split


class DisorderData:
    def __init__(self, data_path, num_classes, random_state):
        self.data_path = data_path
        self.num_classes = num_classes
        self.random_state = random_state

    def prepare_data(self):
        df = pd.read_csv(self.data_path)
        df['Patient Number'] = df['Patient Number'].str[8:].astype(int)

        d = {'Seldom': 1, 'Sometimes': 2, 'Usually': 3, 'Most-Often': 4}

        df['Sadness'] = df['Sadness'].replace(d).astype(int)
        df['Euphoric'] = df['Euphoric'].replace(d).astype(int)
        df['Exhausted'] = df['Exhausted'].replace(d).astype(int)
        df['Sleep dissorder'] = df['Sleep dissorder'].replace(d).astype(int)
        df['Suicidal thoughts'] = df['Suicidal thoughts'].replace('YES ', 'YES')

        d = {'3 From 10': 3, '4 From 10': 4, '6 From 10': 6, '5 From 10': 5, '7 From 10': 7, '8 From 10': 8,
        '9 From 10': 9, '2 From 10': 2, '1 From 10': 1}

        df['Sexual Activity'] = df['Sexual Activity'].replace(d).astype(int)
        df['Concentration'] = df['Concentration'].replace(d).astype(int)
        df['Optimisim'] = df['Optimisim'].replace(d).astype(int)

        yes_no_columns = ['Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect', 'Try-Explanation',
                        'Aggressive Response', 'Ignore & Move-On', 'Nervous Break-down',
                        'Admit Mistakes', 'Overthinking']
        df[yes_no_columns] = df[yes_no_columns].replace({'YES': 1, 'NO': 0})

        df[yes_no_columns] = df[yes_no_columns].apply(pd.to_numeric, errors='coerce')
        df.reset_index(drop=True, inplace=True)

        d = {'Bipolar Type-1': 1, 'Bipolar Type-2': 2, 'Depression': 3, 'Normal': 4}

        df['Expert Diagnose'] = df['Expert Diagnose'].replace(d).astype(int)

        self.df = df

    def get_splitted_data(self):
        self.prepare_data()
        le = LabelEncoder()

        X = self.df.drop(columns='Expert Diagnose').values
        Y = self.df['Expert Diagnose'].values
        Y = le.fit_transform(Y)
        Y = tensorflow.keras.utils.to_categorical(Y, num_classes=self.num_classes)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=self.random_state)
        
        return_labels = ['X_train', 'X_test', 'Y_train', 'Y_test']
        splitted_data = dict(zip(return_labels, [X_train, X_test, Y_train, Y_test]))
        
        return splitted_data
