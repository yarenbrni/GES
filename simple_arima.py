import pandas as pd
import numpy as np
import pyodbc
from statsmodels.tsa.arima.model import ARIMA
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
        print("Veritabanına bağlandı.")
        
        # İlk invertör için verileri çek (sadece 2023 yılı)
        query = """
            SELECT Year, Month, Day, H_i_m
            FROM SolarRadiationData 
            WHERE InverterID = 1 AND Year = 2023
            ORDER BY Year, Month, Day
        """
        
        # Verileri pandas DataFrame'e yükle
        df = pd.read_sql(query, conn)
        print(f"Veri yüklendi. Toplam kayıt sayısı: {len(df)}")
        
        # Günlük ortalama hesapla
        daily_data = df.groupby(['Year', 'Month', 'Day'])['H_i_m'].mean().reset_index()
        daily_data['Date'] = pd.to_datetime(daily_data[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))
        daily_data.set_index('Date', inplace=True)
        ts_data = daily_data['H_i_m']
        
        print("Veri hazırlandı. ARIMA modeli eğitiliyor...")
        
        # ARIMA modelini eğit
        model = ARIMA(ts_data, order=(1,1,1))
        results = model.fit()
        
        # 30 günlük tahmin yap
        forecast = results.forecast(steps=30)
        
        # Sonuçları görselleştir
        plt.figure(figsize=(12, 6))
        plt.plot(ts_data.index, ts_data.values, label='Gerçek Değerler')
        plt.plot(forecast.index, forecast.values, label='Tahminler', linestyle='--')
        plt.title('ARIMA Model Tahminleri (2023)')
        plt.legend()
        plt.savefig('arima_simple_predictions.png')
        plt.close()
        
        print("Tahminler tamamlandı ve grafik kaydedildi.")
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
    finally:
        if conn:
            conn.close()
            print("Bağlantı kapatıldı.")

if __name__ == "__main__":
    main() 