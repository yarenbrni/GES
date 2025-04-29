from flask import Flask, render_template
import pandas as pd
import numpy as np
import pyodbc
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def get_db_connection():
    try:
        print("Veritabanı bağlantısı deneniyor...")
        connection_string = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=(localdb)\\Yaren;"
            "DATABASE=cepdev;"
            "Trusted_Connection=yes;"
        )
        print(f"Bağlantı dizesi: {connection_string}")
        conn = pyodbc.connect(connection_string)
        print("Başarılı bağlantı")
        return conn
    except Exception as e:
        print(f"Veritabanı bağlantı hatası: {str(e)}")
        raise

def train_xgboost_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    model.fit(X_train, y_train)
    return model

def train_lightgbm_model(X_train, y_train):
    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    model.fit(X_train, y_train)
    return model

def train_arima_model(data):
    try:
        # Veriyi numpy array'e dönüştür
        data_array = np.array(data)
        
        # Veri boyutunu kontrol et ve düzelt
        if len(data_array.shape) > 1:
            data_array = data_array.flatten()
        
        print(f"ARIMA için veri boyutu: {data_array.shape}")
        
        # ARIMA modelini oluştur ve eğit
        model = ARIMA(data_array, order=(1,1,1))
        results = model.fit()
        
        return results
    except Exception as e:
        print(f"ARIMA model hatası: {str(e)}")
        return None

@app.route('/')
def index():
    try:
        print("Ana sayfa yükleniyor...")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 2023 yılı verilerini al
        query = """
        SELECT InverterID, CreatedDate, EfficiencyRatio
        FROM SolarRadiationData
        WHERE Year(CreatedDate) = 2023
        ORDER BY CreatedDate
        """
        print("SQL sorgusu çalıştırılıyor...")
        cursor.execute(query)
        data = cursor.fetchall()
        
        if not data:
            print("Veri bulunamadı!")
            raise Exception("2023 yılına ait veri bulunamadı")
            
        print(f"Toplam {len(data)} kayıt bulundu")
        
        # Veriyi DataFrame'e dönüştür
        df = pd.DataFrame(data, columns=['InverterID', 'DateTime', 'EfficiencyRatio'])
        print(f"Benzersiz invertör sayısı: {len(df['InverterID'].unique())}")
        
        # Her invertör için ayrı analiz yap
        results = []
        for inverter_id in df['InverterID'].unique():
            print(f"Invertör {inverter_id} analiz ediliyor...")
            inverter_data = df[df['InverterID'] == inverter_id]
            
            # Günlük ortalama hesapla
            daily_avg = inverter_data.groupby(inverter_data['DateTime'].dt.date)['EfficiencyRatio'].mean()
            print(f"Günlük ortalama veri boyutu: {daily_avg.shape}")
            
            # Veriyi eğitim ve test olarak böl
            train_size = int(len(daily_avg) * 0.8)
            train_data = daily_avg[:train_size]
            test_data = daily_avg[train_size:]
            print(f"Eğitim verisi boyutu: {train_data.shape}, Test verisi boyutu: {test_data.shape}")
            
            # ARIMA modeli
            arima_model = train_arima_model(train_data)
            if arima_model is not None:
                try:
                    arima_predictions = arima_model.forecast(steps=len(test_data))
                    arima_predictions = np.array(arima_predictions)
                    print(f"ARIMA tahmin boyutu: {arima_predictions.shape}")
                except Exception as e:
                    print(f"ARIMA tahmin hatası: {str(e)}")
                    arima_predictions = np.zeros(len(test_data))
            else:
                arima_predictions = np.zeros(len(test_data))
            
            # XGBoost modeli
            X_train = np.arange(len(train_data)).reshape(-1, 1)
            X_test = np.arange(len(test_data)).reshape(-1, 1)
            xgb_model = train_xgboost_model(X_train, train_data)
            xgb_predictions = xgb_model.predict(X_test)
            
            # LightGBM modeli
            lgbm_model = train_lightgbm_model(X_train, train_data)
            lgbm_predictions = lgbm_model.predict(X_test)
            
            # Grafik oluştur
            plt.figure(figsize=(15, 8))
            plt.plot(test_data.values, label='Gerçek Değerler')
            if arima_model is not None:
                plt.plot(arima_predictions, label='ARIMA Tahminleri', linestyle='--')
            plt.plot(xgb_predictions, label='XGBoost Tahminleri', linestyle=':')
            plt.plot(lgbm_predictions, label='LightGBM Tahminleri', linestyle='-.')
            plt.title(f'Invertör {inverter_id} - Model Tahminleri')
            plt.xlabel('Gün')
            plt.ylabel('Verimlilik Oranı')
            plt.legend()
            
            # Grafiği base64 formatına dönüştür
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
            img.seek(0)
            plt.close()
            plot_data = base64.b64encode(img.getvalue()).decode()
            
            # Model performans metrikleri
            arima_metrics = {
                'mse': round(mean_squared_error(test_data, arima_predictions), 4) if arima_model is not None else 'N/A',
                'mae': round(mean_absolute_error(test_data, arima_predictions), 4) if arima_model is not None else 'N/A',
                'r2': round(r2_score(test_data, arima_predictions), 4) if arima_model is not None else 'N/A'
            }
            
            xgb_metrics = {
                'mse': round(mean_squared_error(test_data, xgb_predictions), 4),
                'mae': round(mean_absolute_error(test_data, xgb_predictions), 4),
                'r2': round(r2_score(test_data, xgb_predictions), 4)
            }
            
            lgbm_metrics = {
                'mse': round(mean_squared_error(test_data, lgbm_predictions), 4),
                'mae': round(mean_absolute_error(test_data, lgbm_predictions), 4),
                'r2': round(r2_score(test_data, lgbm_predictions), 4)
            }
            
            results.append({
                'inverter_id': inverter_id,
                'plot': plot_data,
                'arima_metrics': arima_metrics,
                'xgb_metrics': xgb_metrics,
                'lgbm_metrics': lgbm_metrics,
                'total_points': len(daily_avg)
            })
            print(f"Invertör {inverter_id} analizi tamamlandı")
        
        print("Tüm analizler tamamlandı, sayfa render ediliyor...")
        return render_template('index.html', results=results)
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        return render_template('error.html', error=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    app.run(debug=True)