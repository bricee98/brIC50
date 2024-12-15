# IC50 Calculator

A web application for calculating IC50 values from dose-response data using a 4-parameter logistic regression model.

## Features
- Parse tab-delimited concentration-response data
- Fit 4-parameter logistic (4PL) curves
- Calculate IC50 values with confidence intervals
- Generate publication-quality plots
- Support for multiple replicates

## Installation

1. Install Python 3.12.0 from [python.org](https://www.python.org/downloads/release/python-3120/)

2. Clone the repository:
   ```bash
   git clone [your-repo-url]
   cd [repository-name]
   ```

3. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate on Windows
   venv\Scripts\activate

   # Activate on macOS/Linux
   source venv/bin/activate
   ```

4. Install exact package versions:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5001
   ```

## Usage

1. Prepare your data in tab-delimited format:
   ```
   concentration    replicate1    replicate2    replicate3
   0               1000          950           1020
   20              800           790           815
   50              600           630           610
   500             100           120           90
   ```

2. Paste your data into the input field
3. Click "Parse Data" to verify your input
4. Click "Proceed with Analysis" to generate the IC50 curve and statistics

## Development

The application is built with:
- Flask (web framework)
- NumPy (numerical computations)
- SciPy (curve fitting)
- Matplotlib (plotting)