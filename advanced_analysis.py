import pandas as pd
import numpy as np
import pyodbc
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Global bağlantı değişkeni
conn = None

def get_db_connection():
    global conn
    if conn is None:
        connection_string = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=(localdb)\\Yaren;"
            "DATABASE=cepdev;"
            "Trusted_Connection=yes;"
        )
        conn = pyodbc.connect(connection_string)
    return conn

def train_arima_model(data):
    try:
        # Veriyi numpy array'e dönüştür
        data_array = np.array(data, dtype=float)
        
        # ARIMA modelini oluştur ve eğit
        # Daha karmaşık bir model yapısı kullanıyoruz
        model = ARIMA(data_array, order=(2,1,2))
        results = model.fit()
        return results
    except Exception as e:
        print(f"ARIMA model hatası: {str(e)}")
        return None

def train_xgboost_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=500,  # Daha fazla ağaç
        learning_rate=0.01,  # Daha düşük öğrenme oranı
        max_depth=5,  # Daha derin ağaçlar
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50  # Erken durdurma
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model

def train_lightgbm_model(X_train, y_train):
    model = LGBMRegressor(
        n_estimators=500,  # Daha fazla ağaç
        learning_rate=0.01,  # Daha düşük öğrenme oranı
        max_depth=5,  # Daha derin ağaçlar
        num_leaves=16,  # Daha fazla yaprak
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50  # Erken durdurma
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model

def create_features(df):
    """Zaman serisi özellikleri oluşturur"""
    df = df.copy()
    
    # Zaman özellikleri
    df['hour'] = df['DateTime'].dt.hour
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['quarter'] = df['DateTime'].dt.quarter
    df['month'] = df['DateTime'].dt.month
    df['year'] = df['DateTime'].dt.year
    df['dayofyear'] = df['DateTime'].dt.dayofyear
    df['dayofmonth'] = df['DateTime'].dt.day
    df['weekofyear'] = df['DateTime'].dt.isocalendar().week
    
    # Gecikme özellikleri
    for lag in [1, 2, 3, 7, 14]:
        df[f'lag_{lag}'] = df['EfficiencyRatio'].shift(lag)
    
    # Hareketli ortalama özellikleri
    for window in [3, 7, 14, 30]:
        df[f'rolling_mean_{window}'] = df['EfficiencyRatio'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['EfficiencyRatio'].rolling(window=window).std()
    
    return df

def calculate_annual_energy_loss(df, inverter_id):
    """Yıllık enerji kaybını hesaplar"""
    # Verimlilik düşüşünü hesapla
    initial_efficiency = df['EfficiencyRatio'].iloc[0]
    final_efficiency = df['EfficiencyRatio'].iloc[-1]
    efficiency_loss = initial_efficiency - final_efficiency
    
    # Ortalama güç üretimini hesapla
    avg_power = df['Power'].mean()
    
    # Yıllık enerji kaybını hesapla (kWh)
    annual_hours = 8760  # Bir yıldaki saat sayısı
    energy_loss = (avg_power * efficiency_loss * annual_hours) / 1000  # kWh cinsinden
    
    return energy_loss

def calculate_failure_risk(df, model_metrics):
    """Arıza riskini hesaplar"""
    # Risk faktörlerini belirle
    risk_factors = {
        'efficiency_trend': 0.4,  # Verimlilik trendi ağırlığı
        'temperature_variance': 0.3,  # Sıcaklık değişimi ağırlığı
        'model_accuracy': 0.3  # Model doğruluğu ağırlığı
    }
    
    # Verimlilik trendini hesapla
    efficiency_trend = (df['EfficiencyRatio'].iloc[-30:].mean() - 
                       df['EfficiencyRatio'].iloc[-60:-30].mean())
    
    # Sıcaklık değişimini hesapla
    temp_variance = df['Temperature'].std()
    
    # Model doğruluğunu hesapla
    model_accuracy = (model_metrics['r2'] + 1) / 2  # R²'yi 0-1 aralığına normalize et
    
    # Risk skorunu hesapla
    risk_score = (
        efficiency_trend * risk_factors['efficiency_trend'] +
        temp_variance * risk_factors['temperature_variance'] +
        model_accuracy * risk_factors['model_accuracy']
    )
    
    # Risk skorunu yüzdeye çevir
    risk_percentage = min(max(risk_score * 100, 0), 100)
    
    return risk_percentage

def generate_maintenance_recommendations(df, energy_loss, risk_percentage):
    """Bakım önerileri oluşturur"""
    recommendations = []
    
    # Verimlilik kontrolü
    efficiency_drop = (df['EfficiencyRatio'].iloc[0] - df['EfficiencyRatio'].iloc[-1]) / df['EfficiencyRatio'].iloc[0] * 100
    if efficiency_drop > 5:
        recommendations.append(f"Verimlilik oranı son 3 ayda %{efficiency_drop:.1f} düşüş göstermiştir")
    
    # Sıcaklık kontrolü
    temp_threshold = df['Temperature'].mean() + 2 * df['Temperature'].std()
    if df['Temperature'].iloc[-1] > temp_threshold:
        recommendations.append("Sıcaklık değerleri normalin üzerinde seyretmektedir")
    
    # Enerji kaybı kontrolü
    if energy_loss > 1000:  # 1000 kWh üzeri kayıp
        recommendations.append(f"Yıllık enerji kaybı {energy_loss:.0f} kWh seviyesinde, acil bakım gereklidir")
    
    # Risk seviyesine göre bakım önerileri
    if risk_percentage > 70:
        recommendations.append("Yüksek arıza riski nedeniyle acil bakım planlanmalıdır")
    elif risk_percentage > 40:
        recommendations.append("Orta seviye arıza riski mevcut, planlı bakım önerilir")
    
    # Genel bakım önerileri
    if len(recommendations) > 0:
        recommendations.append("Önerilen Bakım: Temizlik ve soğutma sistemi kontrolü")
        recommendations.append("Önerilen Bakım Tarihi: Önümüzdeki 2 hafta içinde")
    
    return recommendations

def main():
    try:
        print("Veritabanı bağlantısı deneniyor...")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Veritabanındaki sütunları kontrol et
        query = """
        SELECT COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'SolarRadiationData'
        """
        print("Veritabanı sütunları kontrol ediliyor...")
        cursor.execute(query)
        columns = [row[0] for row in cursor.fetchall()]
        print(f"Veritabanındaki sütunlar: {columns}")
        
        # Tüm invertörler için verileri al
        query = """
        SELECT DISTINCT InverterID
        FROM SolarRadiationData
        ORDER BY InverterID
        """
        print("\nInvertör listesi alınıyor...")
        cursor.execute(query)
        inverter_ids = [row[0] for row in cursor.fetchall()]
        
        if not inverter_ids:
            print("Invertör bulunamadı!")
            raise Exception("Veritabanında invertör bulunamadı")
            
        print(f"Toplam {len(inverter_ids)} invertör bulundu: {inverter_ids}")
        
        # Her invertör için ayrı analiz yap
        for inverter_id in inverter_ids:
            try:
                print(f"\n{'='*50}")
                print(f"Invertör {inverter_id} analiz ediliyor...")
                
                # İnvertör verilerini al
                query = f"""
                SELECT *
                FROM SolarRadiationData
                WHERE InverterID = ?
                ORDER BY CreatedDate
                """
                cursor.execute(query, inverter_id)
                data = cursor.fetchall()
                
                if not data:
                    print(f"Invertör {inverter_id} için veri bulunamadı!")
                    continue
                
                print(f"Invertör {inverter_id} için {len(data)} kayıt bulundu")
                
                # Veriyi DataFrame'e dönüştür
                df = pd.DataFrame(data, columns=columns)
                
                # Veri tiplerini düzelt
                df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])
                numeric_columns = [col for col in columns if col not in ['CreatedDate', 'InverterID']]
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Eksik verileri kontrol et
                missing_values = df[numeric_columns].isnull().sum()
                if missing_values.any():
                    print("\nEksik Veri Durumu:")
                    for col, count in missing_values.items():
                        if count > 0:
                            print(f"{col}: {count} adet eksik veri")
                    print("Eksik veriler kaldırılıyor...")
                    df = df.dropna(subset=numeric_columns)
                
                # Veri istatistikleri
                print("\nTüm Parametrelerin İstatistikleri:")
                print(df[numeric_columns].describe())
                
                # Yıllara göre veri dağılımı
                print("\nYıllara Göre Veri Dağılımı:")
                yearly_counts = df['CreatedDate'].dt.year.value_counts().sort_index()
                print(yearly_counts)
                
                # Özellik mühendisliği
                df = create_features(df)
                
                # Her parametre için analiz yap
                for target_col in numeric_columns:
                    print(f"\n{'='*50}")
                    print(f"{target_col} Analizi:")
                    
                    # Veriyi eğitim ve test olarak böl
                    train_size = int(len(df) * 0.8)
                    train_data = df[:train_size]
                    test_data = df[train_size:]
                    
                    # Özellikleri hazırla
                    feature_columns = [col for col in df.columns if col not in ['CreatedDate', 'InverterID', target_col]]
                    X_train = train_data[feature_columns]
                    X_test = test_data[feature_columns]
                    y_train = train_data[target_col]
                    y_test = test_data[target_col]
                    
                    # ARIMA modeli
                    print("\nARIMA Modeli Eğitiliyor...")
                    arima_model = train_arima_model(y_train)
                    arima_predictions = None
                    if arima_model is not None:
                        try:
                            arima_predictions = arima_model.forecast(steps=len(y_test))
                            arima_predictions = np.array(arima_predictions, dtype=float)
                            print("ARIMA tahminleri başarıyla oluşturuldu")
                        except Exception as e:
                            print(f"ARIMA tahmin hatası: {str(e)}")
                            arima_predictions = np.zeros(len(y_test), dtype=float)
                    else:
                        arima_predictions = np.zeros(len(y_test), dtype=float)
                    
                    # XGBoost modeli
                    print("\nXGBoost Modeli Eğitiliyor...")
                    xgb_model = train_xgboost_model(X_train, y_train)
                    xgb_predictions = xgb_model.predict(X_test)
                    print("XGBoost tahminleri başarıyla oluşturuldu")
                    
                    # LightGBM modeli
                    print("\nLightGBM Modeli Eğitiliyor...")
                    lgbm_model = train_lightgbm_model(X_train, y_train)
                    lgbm_predictions = lgbm_model.predict(X_test)
                    print("LightGBM tahminleri başarıyla oluşturuldu")
                    
                    # Model performans metrikleri
                    arima_metrics = {
                        'mse': round(mean_squared_error(y_test, arima_predictions), 4) if arima_model is not None else 'N/A',
                        'mae': round(mean_absolute_error(y_test, arima_predictions), 4) if arima_model is not None else 'N/A',
                        'r2': round(r2_score(y_test, arima_predictions), 4) if arima_model is not None else 'N/A'
                    }
                    
                    xgb_metrics = {
                        'mse': round(mean_squared_error(y_test, xgb_predictions), 4),
                        'mae': round(mean_absolute_error(y_test, xgb_predictions), 4),
                        'r2': round(r2_score(y_test, xgb_predictions), 4)
                    }
                    
                    lgbm_metrics = {
                        'mse': round(mean_squared_error(y_test, lgbm_predictions), 4),
                        'mae': round(mean_absolute_error(y_test, lgbm_predictions), 4),
                        'r2': round(r2_score(y_test, lgbm_predictions), 4)
                    }
                    
                    print("\nPerformans Metrikleri:")
                    print("\nARIMA Modeli:")
                    print(f"MSE (Ortalama Kare Hata): {arima_metrics['mse']}")
                    print(f"MAE (Ortalama Mutlak Hata): {arima_metrics['mae']}")
                    print(f"R² (Belirleme Katsayısı): {arima_metrics['r2']}")
                    
                    print("\nXGBoost Modeli:")
                    print(f"MSE (Ortalama Kare Hata): {xgb_metrics['mse']}")
                    print(f"MAE (Ortalama Mutlak Hata): {xgb_metrics['mae']}")
                    print(f"R² (Belirleme Katsayısı): {xgb_metrics['r2']}")
                    
                    print("\nLightGBM Modeli:")
                    print(f"MSE (Ortalama Kare Hata): {lgbm_metrics['mse']}")
                    print(f"MAE (Ortalama Mutlak Hata): {lgbm_metrics['mae']}")
                    print(f"R² (Belirleme Katsayısı): {lgbm_metrics['r2']}")
                    
                    print(f"\n{target_col} analizi tamamlandı")
                    print(f"{'='*50}\n")
                
                # Yıllık enerji kaybını hesapla
                energy_loss = calculate_annual_energy_loss(df, inverter_id)
                
                # Arıza riskini hesapla
                risk_percentage = calculate_failure_risk(df, {
                    'r2': max(arima_metrics['r2'], xgb_metrics['r2'], lgbm_metrics['r2'])
                })
                
                # Bakım önerilerini oluştur
                recommendations = generate_maintenance_recommendations(df, energy_loss, risk_percentage)
                
                # Raporu yazdır
                print(f"\nInvertör {inverter_id} Analiz Raporu:")
                print(f"Yıllık Enerji Kaybı: {energy_loss:.0f} kWh")
                print(f"Arıza Riski: %{risk_percentage:.1f}")
                print("\nBakım Önerileri:")
                for rec in recommendations:
                    print(f"- {rec}")
                
                print(f"\nInvertör {inverter_id} analizi tamamlandı")
                print(f"{'='*50}\n")
            
            except Exception as e:
                print(f"Invertör {inverter_id} analizinde hata: {str(e)}")
                continue
        
        print("Tüm analizler tamamlandı!")
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        import traceback
        print("Hata detayı:")
        print(traceback.format_exc())
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    main() 