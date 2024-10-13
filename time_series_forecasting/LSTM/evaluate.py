import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def make_predictions(model, feature_tensor, target_scaler=None):

    model.eval()
    with torch.no_grad():
        predictions = model(feature_tensor).cpu().numpy()
    if target_scaler!=None:
        predicted_prices = target_scaler.inverse_transform(predictions)
    else:
        predicted_prices = predictions

    return predicted_prices

def evaluate_model(model, feature_tensor, target_tensor, target_scaler):

    pred_target = make_predictions(
        model=model,
        feature_tensor=feature_tensor,
        target_scaler=target_scaler
    )
    
    # Inverse transform the scaled values back to the original scale
    true_price_per_t = target_scaler.inverse_transform(target_tensor.numpy())
    pred_price_per_t = pred_target
    
    # Combine true and predicted prices into a DataFrame
    pred_df = pd.DataFrame(
        np.hstack([true_price_per_t, pred_price_per_t]), columns=['true_price_per_t', 'pred_price_per_t']
    )
    
    # Calculate Mean Squared Error (MSE)
    mse = (mean_squared_error(true_price_per_t, pred_price_per_t))
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root mean Squared Error (MSE): {rmse}")
    
    # Plot the true vs predicted prices
    plt.plot(pred_df['true_price_per_t'], label='Actual Prices')
    plt.plot(pred_df['pred_price_per_t'], label='Predicted Prices')
    plt.legend()
    plt.show()

    return np.round(mse, 2), np.round(rmse, 2)

def evaluate_log_diff_model(model, feature_tensor, target_tensor, target_scaler, df):

    pred_target = make_predictions(
        model=model,
        feature_tensor=feature_tensor,
        target_scaler=target_scaler
    )
    
    pred_log_diff_price_t = pred_target.flatten()
    pred_nrows = pred_log_diff_price_t.shape[0]
    pred_start_row = df.shape[0] - pred_nrows
    pred_df = df.iloc[pred_start_row:, :].copy()
    pred_price_t = np.exp(pred_log_diff_price_t + pred_df['log_price_t_last'].to_numpy())

    true_price_t = np.exp(pred_df['log_diff_price_t'].to_numpy() + pred_df['log_price_t_last'].to_numpy())

    pred_df['price_per_t'] = true_price_t
    pred_df['pred_price_per_t'] = pred_price_t
    
    # Calculate Mean Squared Error (MSE)
    mse = (mean_squared_error(true_price_t, pred_price_t))
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root mean Squared Error (MSE): {rmse}")
    
    # Plot the true vs predicted prices
    plt.plot(pred_df['price_per_t'], label='Actual Prices')
    plt.plot(pred_df['pred_price_per_t'], label='Predicted Prices')
    plt.legend()
    plt.show()

    return np.round(mse, 2), np.round(rmse, 2)
