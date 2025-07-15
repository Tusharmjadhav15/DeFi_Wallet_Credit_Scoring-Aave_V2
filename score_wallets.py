import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------
# Token decimals mapping
# -------------------------
TOKEN_DECIMALS = {
    'USDC': 6,
    'DAI': 18,
    'WETH': 18,
    'WMATIC': 18,
    # Add more as needed
}

# -------------------------
# Load + normalize JSON
# -------------------------
def load_and_normalize(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data, sep='_')
    return df

# -------------------------
# Adjust raw amount for token decimals
# -------------------------
def adjust_amount(row):
    symbol = row['actionData_assetSymbol']
    raw_amount = float(row['actionData_amount'])
    decimals = TOKEN_DECIMALS.get(symbol, 18)
    return raw_amount / (10 ** decimals)

# -------------------------
# Prepare transactions
# -------------------------
def prepare_transactions(df):
    df['token_amount'] = df.apply(adjust_amount, axis=1)
    df['asset_price_usd'] = df['actionData_assetPriceUSD'].astype(float)
    df['amount_usd'] = df['token_amount'] * df['asset_price_usd']
    df = df[df['amount_usd'] > 0].copy()
    df['log_amount_usd'] = np.log1p(df['amount_usd'])
    return df

# -------------------------
# Aggregate wallet-level features
# -------------------------
def aggregate_wallet_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    grouped = df.groupby('userWallet').agg(
        num_deposits=('action', lambda x: (x == 'deposit').sum()),
        num_borrows=('action', lambda x: (x == 'borrow').sum()),
        num_repayments=('action', lambda x: (x == 'repay').sum()),
        num_liquidations=('action', lambda x: (x == 'liquidationcall').sum()),
        total_amount_usd=('amount_usd', 'sum'),
        total_deposited_usd=('amount_usd', lambda x: x[df.loc[x.index, 'action'] == 'deposit'].sum()),
        total_borrowed_usd=('amount_usd', lambda x: x[df.loc[x.index, 'action'] == 'borrow'].sum()),
        total_repaid_usd=('amount_usd', lambda x: x[df.loc[x.index, 'action'] == 'repay'].sum()),
        unique_days_active=('timestamp', lambda x: x.dt.date.nunique())
    ).reset_index()

    grouped['borrow_to_deposit_ratio'] = grouped['total_borrowed_usd'] / grouped['total_deposited_usd'].replace(0, 1)
    grouped['repayment_to_borrow_ratio'] = grouped['total_repaid_usd'] / grouped['total_borrowed_usd'].replace(0, 1)

    return grouped

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # 1. Load & normalize
    json_file = 'user-wallet-transactions.json'
    df_txns = load_and_normalize(json_file)
    print(f"Loaded {len(df_txns)} transactions.")

    # 2. Prepare transactions
    df_txns = prepare_transactions(df_txns)
    print(f"Remaining after zero-check: {len(df_txns)} transactions.")

    # 3. Aggregate wallet features
    wallet_df = aggregate_wallet_features(df_txns)
    print(f"Wallets aggregated: {len(wallet_df)}")

    # 4. Pseudo-labels
    wallet_df['risk_label'] = 0
    wallet_df.loc[wallet_df['num_liquidations'] > 0, 'risk_label'] = 1
    wallet_df.loc[wallet_df['borrow_to_deposit_ratio'] > 1.5, 'risk_label'] = 1
    wallet_df.loc[wallet_df['repayment_to_borrow_ratio'] < 0.5, 'risk_label'] = 1
    wallet_df['credit_score_target'] = (1 - wallet_df['risk_label']) * 1000

    # 5. Model training
    features = [
        'num_deposits', 'num_borrows', 'num_repayments', 'num_liquidations',
        'total_deposited_usd', 'total_borrowed_usd', 'total_repaid_usd',
        'unique_days_active', 'borrow_to_deposit_ratio', 'repayment_to_borrow_ratio'
    ]
    X = wallet_df[features].fillna(0)
    y = wallet_df['credit_score_target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test).clip(0, 1000)
    r2 = model.score(X_test, y_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))

    print(f"R^2 score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # 6. Score all wallets
    wallet_df['credit_score'] = model.predict(X_scaled).clip(0, 1000)

    # 7. Save final scores
    wallet_df[['userWallet', 'credit_score']].to_csv('wallet_scores.csv', index=False)
    print("Saved wallet_scores.csv")
