# House Price Predictor 🏠
A machine learning solution to predict residential real estate prices using Linear Regression.

## 🚀 Project Overview
This project implements Linear Regression from scratch using gradient descent to predict house prices. Built as part of my **AI Engineering Portfolio**, demonstrating:
- Custom implementation of gradient descent
- Feature engineering and selection
- Model evaluation and visualization
- Achieving 84% R² score

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn
- **Environment:** Jupyter Lab & PyCharm
- **Version Control:** Git/GitHub

## 📁 Project Structure
```
house-price-predictor/
├── notebooks/          # Jupyter notebooks for EDA and experiments
│   └── 01_eda.ipynb   # Complete analysis and model training
├── src/               # Source code
│   └── linear_regression.py  # Custom LinearRegression class
├── data/              # Dataset folder (train.csv, test.csv)
├── .gitignore
├── requirements.txt
└── README.md
```

## 📊 Dataset
This project uses the **Kaggle House Prices - Advanced Regression Techniques** dataset.

**Download Instructions:**
1. Go to: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
2. Download `train.csv` and `test.csv`
3. Place both files in the `data/` folder

**Note:** CSV files are not included in this repository due to size. Download them locally before running the notebooks.

## ⚙️ Setup Instructions
1. Clone the repository:
```bash
   git clone https://github.com/galileo-gal/house-price-predictor.git
   cd house-price-predictor
```

2. Create virtual environment:
```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Download dataset (see Dataset section above)

5. Run Jupyter:
```bash
   jupyter lab
```

6. Open `notebooks/01_eda.ipynb`

## 📈 Results (Day 3)
- **Model:** Linear Regression (from scratch)
- **Training R²:** 0.8449
- **Validation R²:** 0.8484
- **Features:** 7 numerical features
- **Convergence:** 99.98% cost reduction in 1000 iterations

## 🎯 Next Steps
- [ ] Add polynomial features
- [ ] Implement L1/L2 regularization
- [ ] Compare with sklearn
- [ ] Write Medium article

## 👤 Author
AI Engineering Student

## 📄 License
MIT License