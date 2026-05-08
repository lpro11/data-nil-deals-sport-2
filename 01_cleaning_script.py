import pandas as pd
import glob
import os
import warnings

# Mapping inconsistent school headers to a standard set [1, 3]
header_mapping = {
    'Transaction Date': 'date', 'Date of Deal': 'date', 'Date': 'date', 'Submitted Date': 'date', 'Transaction Date': 'date',
    'Amount': 'amount', 'Total Compensation': 'amount', 'Cost': 'amount', 'Proposed Payment Amount (i.e. $5 per post, $500 per appearance, etc.):': 'amount', 'Total Value of Reported Deals': 'amount', 'Value': 'amount', 'Total NIL': 'amount',
    'Sport Name': 'sport', 'Athletic Team': 'sport', 'Sport': 'sport', 'Sport Type': 'sport', 'Team': 'sport',
    'Description': 'deal_description', 'Transaction Type': 'deal_description', 'Brief Description': 'deal_description', 'Disclosure Type/Transaction Type': 'deal_description', 'Type of Endorsement:': 'deal_description', 'Transactions': 'deal_description', 'Notes': 'deal_description'
}

def clean_nil_data():
    # Identify all raw CSV files in the data/raw folder [5-11]
    raw_path = 'data/raw/*.csv'
    all_files = glob.glob(raw_path)
    clean_dfs = []

    for file in all_files:
        school_name = os.path.basename(file).replace('.csv', '')
        if school_name in ['fresnostate1', 'sandiegostate1']:
            print(f"Skipping {school_name}: manually excluded")
            continue
        # Read the school file (e.g., ucla.csv, ucdavis.csv) [10, 11]
        df = pd.read_csv(file)
        df = df.rename(columns=header_mapping)
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns if any
        
        # Check for required columns
        required_cols = ['amount']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Skipping {school_name}: missing required columns {missing}")
            continue
        
        # Add school name to control for school-level differences [4]
        df['school'] = school_name
        
        # Identify "Social Media" deals by searching for keywords [3, 4]
        # This creates your primary treatment variable
        if 'deal_description' in df.columns:
            df['is_social_media'] = df['deal_description'].str.contains(
                'social media|instagram|tiktok|post|tweet', case=False, na=False
            ).astype(int)
        else:
            df['deal_description'] = 'Other'
            df['is_social_media'] = 0 # Default to 0 if description is missing
        
        # Ensure blank deal_descriptions are set to 'Other'
        df['deal_description'] = df['deal_description'].fillna('Other').replace('', 'Other')
        
        # Special handling for ucsandiego1: if Notes blank (now 'Other'), assume Social Media
        if school_name == 'ucsandiego1':
            df.loc[df['deal_description'] == 'Other', 'deal_description'] = 'Social Media'
        
        if 'sport' not in df.columns:
            df['sport'] = 'Other'
        
        # Standardise amount: convert to numbers, handling missing or 'in-kind' values [2]
        if 'amount' in df.columns:
            df['amount'] = df['amount'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        else:
            df['amount'] = 0

        # Try to infer/standardise dates from any known date columns
        date_candidates = [
            'date', 'Submitted Date', 'Last Updated', 'Decision Date',
            'Transaction Date', 'Upload Date', 'Date(s) of activity (i.e. what date range will you post the tweet(s), when will the appearance be?)',
            'Date(s) of publication (i.e. what date range will you post the tweet, when will the appearance be?)',
            'Created Date', 'reporting date', 'Date/Time'
        ]

        if 'date' not in df.columns:
            df['date'] = pd.NaT

        if df['date'].isna().all():
            for candidate in date_candidates:
                if candidate in df.columns:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UserWarning)
                        inferred = pd.to_datetime(df[candidate], errors='coerce')
                    if inferred.notna().any():
                        df['date'] = inferred
                        break

        # Convert date to year, assign random 2021-2024 if missing
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        import numpy as np
        np.random.seed(42)
        df['year'] = df['year'].apply(lambda x: np.random.choice([2021, 2022, 2023, 2024]) if pd.isna(x) else x)
        
        # Special handling for Year column
        if 'Year' in df.columns and df['year'].isna().all():
            df['year'] = df['Year'].str.split('-').str[0].astype(int)
        
        df['date'] = df['year'].astype(int)

        # Select standard columns for the final dataset
        cols = ['school', 'date', 'amount', 'sport', 'deal_description', 'is_social_media']
        clean_dfs.append(df[[c for c in cols if c in df.columns]])

    # Combine all schools into one master file [1]
    combined = pd.concat(clean_dfs, ignore_index=True)
    print(f"Combined data has {len(combined)} rows before filtering.")
    
    # Filter for amount > 0
    combined = combined[combined['amount'] > 0]
    print(f"After amount filter, {len(combined)} rows.")
    
    # Standardize sport labels
    sport_mapping = {
        # Basketball
        'Men\'s Basketball': 'Basketball',
        'Women\'s Basketball': 'Basketball',
        'Basketball (Mens)': 'Basketball',
        'Basketball (Womens)': 'Basketball',
        'MBB': 'Basketball',
        'WBB': 'Basketball',
        # Volleyball
        'Men\'s Volleyball': 'Volleyball',
        'Women\'s Volleyball': 'Volleyball',
        'Volleyball (Mens)': 'Volleyball',
        'Volleyball (Womens)': 'Volleyball',
        'MVB': 'Volleyball',
        'WVB': 'Volleyball',
        'Beach VB': 'Beach Volleyball',
        'Beach Volleyball': 'Beach Volleyball',
        'Volleyball (Beach)': 'Beach Volleyball',
        # Water Polo
        'Men\'s Water Polo': 'Water Polo',
        'Women\'s Water Polo': 'Water Polo',
        'Water Polo (Mens)': 'Water Polo',
        'Water Polo (Womens)': 'Water Polo',
        'MWP': 'Water Polo',
        # Track and Field
        'Men\'s Outdoor Track & Field': 'Track and Field',
        'Track and Field (Men\'s)': 'Track and Field',
        'Track and Field (Mens)': 'Track and Field',
        'Track and Field (Womens)': 'Track and Field',
        'MTR/XC': 'Track and Field',
        'WTR/XC': 'Track and Field',
        'Men\'s Track': 'Track and Field',
        'Women\'s Track': 'Track and Field',
        'Track and Field': 'Track and Field',
        # Soccer
        'Men\'s Soccer': 'Soccer',
        'Women\'s Soccer': 'Soccer',
        'Soccer (Mens)': 'Soccer',
        'Soccer (Womens)': 'Soccer',
        'MSW': 'Soccer',
        'WSW': 'Soccer',
        'MSO': 'Soccer',
        'WSO': 'Soccer',
        # Golf
        'Golf (Mens)': 'Golf',
        'Golf (Womens)': 'Golf',
        'Women\'s Golf': 'Golf',
        # Swimming/Diving
        'Swimming/Diving (Mens)': 'Swimming/Diving',
        'Swimming/Diving (Womens)': 'Swimming/Diving',
        # Tennis
        'Tennis (Mens)': 'Tennis',
        'Tennis (Womens)': 'Tennis',
        'MTE': 'Tennis',
        'WTE': 'Tennis',
        'Women\'s Tennis': 'Tennis',
        # Rowing
        'Rowing (Mens)': 'Rowing',
        'Rowing (Womens)': 'Rowing',
        'MRO': 'Rowing',
        'WRO': 'Rowing',
        # Gymnastics
        'Gymnastics (Mens)': 'Gymnastics',
        'Gymnastics (Womens)': 'Gymnastics',
        'Gymnastics': 'Gymnastics',
        # Lacrosse
        'Lacrosse': 'Lacrosse',
        'Lacrosse (Womens)': 'Lacrosse',
        # Cross Country
        'Cross Country': 'Cross Country',
        'Cross Country (Womens)': 'Cross Country',
        # Rugby
        'Rugby (Mens)': 'Rugby',
        # Field Hockey
        'Field Hockey': 'Field Hockey',
        'Women\'s Field Hockey': 'Field Hockey',
        # Softball
        'Softball': 'Softball',
        'WSB': 'Softball',  # Assuming Women's Softball
        # Other
        'MBA': 'Baseball',  # Assuming Men's Baseball
        'Footballl': 'Football',  # Typo
        'Men\'s football': 'Football',
        'General': 'Other',
        'Business': 'Other',
        'Other': 'Other',
        # Keep as is if not mapped
    }
    combined['sport'] = combined['sport'].map(sport_mapping).fillna(combined['sport'])
    
    # Save the cleaned file
    combined.to_csv('data/clean/cleaned_nil.csv', index=False)
    print("Step 1 Complete: cleaned_nil.csv created.")

if __name__ == "__main__":
    clean_nil_data()
