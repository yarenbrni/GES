<<<<<<< HEAD
=======
# GES
>>>>>>> 389b681a2649b0a8fab7d99b9bc39a7e1f6dc59e
# PVsyst SQL Server Bağlantı Projesi

Bu proje, PVsyst'ten alınan güneş ışınımı verilerini SQL Server'a aktarmak ve yönetmek için kullanılır.

## Gereksinimler

- Python 3.8 veya üzeri
- SQL Server
- ODBC Driver for SQL Server

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. `.env` dosyasını düzenleyin:
- SQL_SERVER: SQL Server adınız
- SQL_DATABASE: Veritabanı adınız
- SQL_USERNAME: Kullanıcı adınız
- SQL_PASSWORD: Şifreniz

## Kullanım

1. Programı çalıştırın:
```bash
python pvsyst_sql_connection.py
```

2. Program otomatik olarak:
- SQL Server'a bağlanır
- Gerekli tabloları oluşturur
- CSV dosyasındaki verileri SQL Server'a aktarır
- 2023 yılı verilerini örnek olarak gösterir

## Özellikler

- CSV dosyasından veri aktarımı
- Yıl ve aya göre veri sorgulama
- Otomatik tablo oluşturma
- Hata yönetimi ve loglama

## Veri Yapısı

SolarRadiationData tablosu şu alanları içerir:
- ID (Otomatik artan)
- Year (Yıl)
- Month (Ay)
- H_h_m (Yatay düzlemdeki güneş ışınımı)
- H_i_opt_m (Optimal eğimli düzlemdeki güneş ışınımı)
- H_i_m (Eğimli düzlemdeki güneş ışınımı)
- Hb_n_m (Normal düzlemdeki direkt güneş ışınımı)
- T2m (Sıcaklık verisi)
<<<<<<< HEAD
- CreatedDate (Kayıt oluşturma tarihi) 
=======
- CreatedDate (Kayıt oluşturma tarihi) 
>>>>>>> 389b681a2649b0a8fab7d99b9bc39a7e1f6dc59e
