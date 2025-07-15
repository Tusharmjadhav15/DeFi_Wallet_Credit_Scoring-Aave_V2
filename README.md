# DeFi_Wallet_Credit_Scoring-Aave_V2
This project builds a robust machine learning pipeline that generates a credit score (0–1000) for each wallet address on the Aave V2 protocol, based purely on historical transaction behavior

- **Higher scores** → Responsible, reliable usage  
- **Lower scores** → Risky, bot-like, or exploitative usage

The goal is to help DeFi protocols better evaluate user risk profiles in a fully on-chain, transparent way.

---

## ✅ Data Source

- Sample **~100K raw, transaction-level JSON** from Aave V2.
- Actions include:
  - `deposit`
  - `borrow`
  - `repay`
  - `redeemunderlying`
  - `liquidationcall`

---

## ✅ Project Architecture
Raw JSON

[Flatten + Normalize]

[Remove zero-value txns + Log-transform]

[Aggregate wallet-level features]

[Pseudo-label risk using simple rules]

[Train ML regressor → predict credit score]

[Clamp scores to 0–1000, save output]


---

## ✅ Methodology

### 1️⃣ Normalize & Clean
- Used `pandas.json_normalize()` to flatten nested `actionData`.
- Converted token `amount` fields to **human-readable** amounts using token decimals.
- Computed `amount_usd = token_amount × asset_price_usd`.
- Removed zero-value transactions.

---

### 2️⃣ Feature Engineering
Each wallet was aggregated into:
- **Counts**: `num_deposits`, `num_borrows`, `num_repayments`, `num_liquidations`
- **Totals**: `total_deposited_usd`, `total_borrowed_usd`, `total_repaid_usd`
- **Ratios**: `borrow_to_deposit_ratio`, `repayment_to_borrow_ratio`
- **Activity**: `unique_days_active`

---

### 3️⃣ Pseudo-Label Risk Rules
We use simple domain rules to create pseudo-labels:
- Liquidations present? → risky
- High `borrow_to_deposit_ratio` (>1.5) → risky
- Low `repayment_to_borrow_ratio` (<0.5) → risky

**Pseudo-label:**  
- Risky wallets → `0`
- Healthy wallets → `1000`

---

### 4️⃣ Machine Learning
- Scaled features with `StandardScaler`.
- Trained a `GradientBoostingRegressor` to predict pseudo credit scores.
- Evaluated performance:
  - **R²**: 0.9661
  - **RMSE**: 78.99
  - **MAE**: 15.88

This shows that the features strongly replicate the pseudo-risk rules.

---

### 5️⃣ Output
- Final scores are **clamped to 0–1000**.
- One-step script generates `wallet_scores.csv` with:
  - `userWallet`
  - `credit_score`

---

## ✅ How to Run

### 1. Install dependencies
pip install pandas scikit-learn matplotlib

### 2. Run the scoring pipeline
python score_wallets.py

### 3. Output
wallet_scores.csv

### ✅ Deliverables
score_wallets.py → One-step pipeline: load raw JSON → score wallets.

wallet_scores.csv → Final output.

README.md → Explains method, architecture, and flow.

analysis.md → Score distribution & wallet behavior insights.

### ✅ Next Steps & Extensibility
Use real default/outcome data instead of pseudo-labels.

Add more wallet-level behavioral features.

Test with anomaly detection for bots & exploits.

### ✅ Repo Structure
.

├── score_wallets.py

├── user_transactions.json

├── wallet_scores.csv

├── README.md

└── analysis.md


