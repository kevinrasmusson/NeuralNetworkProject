from dataset import load_data
from model import create_model

if __name__ == '__main__':
    df = load_data('text-ds.csv')  
    print(df["text_clean"].iloc[0])
    #model = create_model(vocab_size=1000)  # Example values, replace with actual values
