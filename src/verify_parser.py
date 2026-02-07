
from parser import WhatsAppParser

def test_parser():
    file_path = '../data/WhatsApp Chat with gg bOys.txt'
    parser = WhatsAppParser(file_path)
    df = parser.parse()
    
    print("Head of DataFrame:")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nCheck for nulls:")
    print(df.isnull().sum())
    
    # Check if any legitimate messages were missed/misparsed as NaT
    if df['DateTime'].isnull().any():
        print("\nWARNING: Some DateTimes could not be parsed.")
        print(df[df['DateTime'].isnull()])

if __name__ == "__main__":
    test_parser()
