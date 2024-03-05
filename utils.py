import  numpy as np
import pickle
import pandas as pd
import config
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


class CarPriceModel():
    def __init__(self):
        self.label_encoder = self.load_label_encoder()
        self.onehot_encoder = self.load_onehot_encoder()
        self.pca_model = self.load_pca_model()
        self.model = self.load_model()

        
    def load_label_encoder(self):
        with open(config.LABEL_FILE_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        return label_encoder

    def load_onehot_encoder(self):
        with open(config.ONEHOT_FILE_PATH, 'rb') as f:
            onehot_encoder = pickle.load(f)
        return onehot_encoder

    def load_pca_model(self):
        with open(config.PCA_FILE_PATH, 'rb') as f:
            pca_model = pickle.load(f)
        return pca_model

    def load_model(self):
        with open(config.MODEL_FILE_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    

    def preprocess_data(self, df):
        # Apply one-hot encoding
        one_hot_cols = ['carbody', 'drivewheel', 'enginetype', 'cylindernumber', 'fuelsystem']
        df_encoded = pd.DataFrame(self.onehot_encoder.transform(df[one_hot_cols]).toarray(), columns=self.onehot_encoder.get_feature_names_out(one_hot_cols))
        df = pd.concat([df, df_encoded], axis=1)
        df = df.drop(one_hot_cols, axis=1)
        
        # Apply label encoding
        label_cols = ['fueltype', 'aspiration', 'doornumber', 'enginelocation']
        for col in label_cols:
            self.label_encoder.fit(df[col])
            df[col] = self.label_encoder.transform(df[col])
        return df
    
    def apply_pca(self, df):
        df_pca_array = self.pca_model.transform(df)
        pca_columns = [f"PC{i}" for i in range(1, self.pca_model.n_components_ + 1)]
        df_pca_array = pd.DataFrame(df_pca_array, columns=pca_columns)
        return df_pca_array
    
    def predict_price(self, df):
        # Preprocess data
        df_preprocessed = self.preprocess_data(df)

        # Apply PCA
        df_pca_array = self.apply_pca(df_preprocessed)

        # Make prediction
        predicted_price = self.model.predict(df_pca_array)[0]
        return predicted_price