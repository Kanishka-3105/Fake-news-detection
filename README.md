# Crunchbase Lead Generator

## Overview
Crunchbase Lead Generator is a Python-based tool that extracts company data from Crunchbase, performs exploratory data analysis (EDA), and uses machine learning to predict potential investment leads. If web scraping fails, the tool automatically generates sample data to ensure functionality.

## Features
- **Web Scraping:** Uses Selenium and BeautifulSoup to extract company data from Crunchbase.
- **Fallback Mechanism:** If scraping fails, sample data is generated.
- **Exploratory Data Analysis (EDA):** Visualizes industry trends, funding distributions, and company insights using Seaborn and Matplotlib.
- **Machine Learning Model:** Trains a Random Forest Classifier to predict companies likely to raise funds.
- **Lead Prediction:** Identifies high-potential investment leads based on the trained model.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/crunchbase-lead-generator.git
   cd crunchbase-lead-generator
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Dependencies
The project requires the following libraries:
- `pandas`
- `numpy`
- `selenium`
- `beautifulsoup4`
- `seaborn`
- `matplotlib`
- `scikit-learn`

Install them using:
```sh
pip install pandas numpy selenium beautifulsoup4 seaborn matplotlib scikit-learn
```

## Usage
Run the script:
```sh
python crunchbase_lead_generator.py
```

### Main Steps
1. **Scrape Crunchbase Data:** Attempts to fetch company data from Crunchbase.
2. **Generate Sample Data (if needed):** If scraping fails, synthetic data is created.
3. **Perform EDA:** Generates visualizations for key insights.
4. **Train Model:** Trains a machine learning model to predict potential investment leads.
5. **Identify Potential Leads:** Lists companies likely to raise funds.




## Future Enhancements
- Implement Selenium automation for dynamic content.
- Enhance the model with additional company metrics.
- Integrate an API for real-time Crunchbase data retrieval.


