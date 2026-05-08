#As the research question focuses on the average value for sports teams between 2021 and 2024, this script aggregates individual payments up to the sport-team level
import pandas as pd

def aggregate_to_team_level():
    # Load the standardised data created in Step 1
    df = pd.read_csv('data/clean/cleaned_nil.csv')
    
    # Date is already the year
    df['year'] = df['date']
    
    # Group by School, Sport, Year, and Deal Type
    # This calculates the average transaction value per sport team [3, 4]
    final_analysis = df.groupby(['school', 'sport', 'year', 'is_social_media']).agg({
        'amount': 'mean'
    }).reset_index()
    
    # Rename for clarity as the 'outcome variable'
    final_analysis.rename(columns={'amount': 'avg_transaction_value'}, inplace=True)
    
    # Save the final analysis-ready dataset for Part B submission
    final_analysis.to_csv('data/clean/nil_merged_analysis.csv', index=False)
    print("Step 2 Complete: nil_merged_analysis.csv is ready for your empirical analysis.")

if __name__ == "__main__":
    aggregate_to_team_level()
