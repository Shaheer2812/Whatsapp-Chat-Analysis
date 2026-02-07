
import re
import pandas as pd
from datetime import datetime

class WhatsAppParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []

    def parse(self):
        """
        Parses the WhatsApp chat file and returns a pandas DataFrame.
        """
        # Regex patterns to try
        # Pattern 1: [dd/mm/yy, HH:MM:SS] Author: Message (iOS/standard export)
        # Pattern 2: dd/mm/yyyy, HH:MM - Author: Message (Android/other export)
        patterns = [
            r'^\[(\d{2}/\d{2}/\d{2}, \d{2}:\d{2}:\d{2})\] (.*?): (.*)$',
            r'^(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - (.*?): (.*)$'
        ]
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        message_buffer = [] 
        date, time, author = None, None, None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = None
            for p in patterns:
                match = re.match(p, line)
                if match:
                    break
            
            if match:
                # If there's a previous message in the buffer, save it
                if author:
                    self.data.append([date_str, author, ' '.join(message_buffer)])
                
                # Start new message
                message_buffer = []
                date_str = match.group(1)
                author = match.group(2)
                message = match.group(3)
                message_buffer.append(message)
            else:
                # If no match, it's a continuation of the previous message
                if author:
                    message_buffer.append(line)

        # Append the last message
        if author:
            self.data.append([date_str, author, ' '.join(message_buffer)])

        df = pd.DataFrame(self.data, columns=['DateTime', 'Author', 'Message'])
        
        # Convert DateTime to datetime objects
        # We need to handle multiple formats now
        df = pd.DataFrame(self.data, columns=['DateTime', 'Author', 'Message'])
        
        # Vectorized date parsing - much faster
        # Try primary format first
        df['temp_date'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y, %H:%M', errors='coerce')
        
        # Try secondary format for rows that failed
        mask = df['temp_date'].isna()
        if mask.any():
            df.loc[mask, 'temp_date'] = pd.to_datetime(df.loc[mask, 'DateTime'], format='%d/%m/%y, %H:%M:%S', errors='coerce')
            
        df['DateTime'] = df['temp_date']
        df.drop(columns=['temp_date'], inplace=True)

        return df

if __name__ == "__main__":
    # Test with dummy data
    parser = WhatsAppParser('../data/_chat.txt')
    df = parser.parse()
    print(df.head())
    print(f"Total messages: {len(df)}")
