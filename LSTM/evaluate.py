import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def make_predictions(model, feat_X_tensor, target_scaler=None):

    model.eval()
    with torch.no_grad():
        predictions = model(feat_X_tensor).cpu().numpy()
    if target_scaler!=None:
        predicted_prices = target_scaler.inverse_transform(predictions)
    else:
        predicted_prices = predictions

    return predicted_prices

def evaluate_training(model, X_train_tensor, y_train_tensor, target_scaler):

    pred_y_train = make_predictions(
        model=model,
        feat_X_tensor=X_train_tensor,
        target_scaler=target_scaler
    )
    
    # Inverse transform the scaled values back to the original scale
    true_price_per_t = target_scaler.inverse_transform(y_train_tensor.numpy())
    pred_price_per_t = pred_y_train
    
    # Combine true and predicted prices into a DataFrame
    train_pred_df = pd.DataFrame(
        np.hstack([true_price_per_t, pred_price_per_t]), columns=['true_price_per_t', 'pred_price_per_t']
    )
    
    # Calculate Mean Squared Error (MSE)
    mse = (mean_squared_error(true_price_per_t, pred_price_per_t))
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root mean Squared Error (MSE): {rmse}")
    
    # Plot the true vs predicted prices
    plt.plot(train_pred_df['true_price_per_t'], label='Actual Prices')
    plt.plot(train_pred_df['pred_price_per_t'], label='Predicted Prices')
    plt.legend()
    plt.show()

    return np.round(mse, 2), np.round(rmse, 2), 
