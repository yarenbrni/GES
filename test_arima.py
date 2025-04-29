import pandas as pd
import numpy as np
import pyodbc
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def main():
    # SQL Server bağlantısı
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=(localdb)\\Yaren;"
        "DATABASE=cepdev;"
        "Trusted_Connection=yes;"
    )
    
    try:
        # Bağlantıyı kur
        conn = pyodbc.connect(connection_string)
        
        # İlk invertör için verileri çek
        query = """
            SELECT Year, Month, Day, Hour, H_i_m
            FROM SolarRadiationData 
            WHERE InverterID = 1
            ORDER BY Year, Month, Day, Hour
        """
        
        # Verileri pandas DataFrame'e yükle
        df = pd.read_sql(query, conn)
        
        # Günlük ortalama hesapla
        daily_data = df.groupby(['Year', 'Month', 'Day'])['H_i_m'].mean().reset_index()
        daily_data['Date'] = pd.to_datetime(daily_data[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))
        daily_data.set_index('Date', inplace=True)
        ts_data = daily_data['H_i_m']
        
        # Veriyi train ve test olarak böl
        train_size = int(len(ts_data) * 0.8)
        train_data = ts_data[:train_size]
        test_data = ts_data[train_size:]
        
        # ARIMA modelini eğit
        print("ARIMA modeli eğitiliyor...")
        model = ARIMA(train_data, order=(1,1,1))
        results = model.fit()
        
        # Tahminler
        print("Tahminler yapılıyor...")
        predictions = results.forecast(steps=len(test_data))
        
        # Model performansını değerlendir
        mse = mean_squared_error(test_data, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_data, predictions)
        
        print("\nARIMA Model Performansı:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Tahmin grafiği
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data.values, label='Gerçek Değerler')
        plt.plot(test_data.index, predictions, label='ARIMA Tahminleri', linestyle='--')
        plt.title('ARIMA Model Tahminleri')
        plt.legend()
        plt.savefig('arima_test_predictions.png')
        plt.close()
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
    finally:
        if conn:
            conn.close()
            print("Bağlantı kapatıldı.")

if __name__ == "__main__":
    main() 