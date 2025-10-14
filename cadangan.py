import pickle
import os
import pandas as pd
import numpy as np
import tensorflow as tf
# Import 'request' untuk membaca parameter dari URL
from flask import Flask, render_template, url_for, jsonify, request
from datetime import timedelta
import holidays
from sklearn.model_selection import train_test_split
import locale

try:
    locale.setlocale(locale.LC_TIME, 'id_ID.UTF-8')
except locale.Error:
    print("Locale id_ID.UTF-8 tidak tersedia, menggunakan locale default.")

@tf.keras.utils.register_keras_serializable()
class ALSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (self.units, self.units)
        self.output_size = self.units
        super(ALSTMCell, self).__init__(**kwargs)
    def get_config(self):
        config = super(ALSTMCell, self).get_config()
        config.update({'units': self.units})
        return config
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W_i=self.add_weight(shape=(input_dim, self.units), name='W_i', initializer='glorot_uniform')
        self.U_i=self.add_weight(shape=(self.units, self.units), name='U_i', initializer='orthogonal')
        self.b_i=self.add_weight(shape=(self.units,), name='b_i', initializer='zeros')
        self.W_f=self.add_weight(shape=(input_dim, self.units), name='W_f', initializer='glorot_uniform')
        self.U_f=self.add_weight(shape=(self.units, self.units), name='U_f', initializer='orthogonal')
        self.b_f=self.add_weight(shape=(self.units,), name='b_f', initializer='zeros')
        self.W_o=self.add_weight(shape=(input_dim, self.units), name='W_o', initializer='glorot_uniform')
        self.U_o=self.add_weight(shape=(self.units, self.units), name='U_o', initializer='orthogonal')
        self.b_o=self.add_weight(shape=(self.units,), name='b_o', initializer='zeros')
        self.W_c=self.add_weight(shape=(input_dim, self.units), name='W_c', initializer='glorot_uniform')
        self.U_c=self.add_weight(shape=(self.units, self.units), name='U_c', initializer='orthogonal')
        self.b_c=self.add_weight(shape=(self.units,), name='b_c', initializer='zeros')
        self.W_a=self.add_weight(shape=(input_dim, input_dim), name='W_a', initializer='glorot_uniform')
        self.U_a=self.add_weight(shape=(self.units, input_dim), name='U_a', initializer='orthogonal')
        self.b_a=self.add_weight(shape=(input_dim,), name='b_a', initializer='zeros')
        self.built = True
    def call(self, inputs, states):
        h_tm1, c_tm1 = states[0], states[1]
        attention_scores=tf.nn.tanh(tf.matmul(inputs, self.W_a) + tf.matmul(h_tm1, self.U_a) + self.b_a)
        attention_weights=tf.nn.softmax(attention_scores, axis=-1)
        context_vector=inputs * attention_weights
        i=tf.sigmoid(tf.matmul(context_vector, self.W_i) + tf.matmul(h_tm1, self.U_i) + self.b_i)
        f=tf.sigmoid(tf.matmul(context_vector, self.W_f) + tf.matmul(h_tm1, self.U_f) + self.b_f)
        o=tf.sigmoid(tf.matmul(context_vector, self.W_o) + tf.matmul(h_tm1, self.U_o) + self.b_o)
        c_=tf.nn.tanh(tf.matmul(context_vector, self.W_c) + tf.matmul(h_tm1, self.U_c) + self.b_c)
        c_t=f * c_tm1 + i * c_
        h_t=o * tf.nn.tanh(c_t)
        return h_t, [h_t, c_t]

app = Flask(__name__)

obat_info = {
    'ibuprofen': {'id': 'ibuprofen', 'name': 'Ibuprofen', 'file': 'ibuprofen.xlsx', 'description': 'Obat untuk meredakan nyeri.', 'icon': 'fas fa-pills'},
    'sanmol': {'id': 'sanmol', 'name': 'Sanmol', 'file': 'sanmol.xlsx', 'description': 'Obat untuk meredakan demam.', 'icon': 'fas fa-thermometer-half'},
    'ranitidin': {'id': 'ranitidin', 'name': 'Ranitidin', 'file': 'ranitidin.xlsx', 'description': 'Obat untuk masalah lambung.', 'icon': 'fas fa-stomach'},
    'optalvit': {'id': 'optalvit', 'name': 'Optalvit', 'file': 'optalvit.xlsx', 'description': 'Suplemen dan vitamin.', 'icon': 'fas fa-capsules'}
}

def engineer_features(df_raw):
    df = df_raw.copy()
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
    df.sort_values('Tanggal', inplace=True)
    df = df.groupby('Tanggal', as_index=False)['Jumlah Terjual'].sum()
    full_dates = pd.date_range(start=df['Tanggal'].min(), end=df['Tanggal'].max(), freq='D')
    df = df.set_index('Tanggal').reindex(full_dates).fillna(0).rename_axis('Tanggal').reset_index()
    df['Bulan'] = df['Tanggal'].dt.month; df['Hari'] = df['Tanggal'].dt.day; df['Hari_Ke'] = df['Tanggal'].dt.dayofyear; df['day_of_week'] = df['Tanggal'].dt.dayofweek; df['is_weekend'] = df['Tanggal'].dt.dayofweek.isin([5, 6]).astype(int); df['is_start_of_month'] = df['Tanggal'].dt.is_month_start.astype(int); df['is_end_of_month'] = df['Tanggal'].dt.is_month_end.astype(int); df['week_of_year'] = df['Tanggal'].dt.isocalendar().week.astype(int); df['quarter'] = df['Tanggal'].dt.quarter
    indonesia_holidays = holidays.Indonesia(years=list(range(df['Tanggal'].min().year, df['Tanggal'].max().year + 1)))
    holiday_dates = set(indonesia_holidays.keys())
    df['is_holiday'] = df['Tanggal'].dt.date.isin(holiday_dates).astype(int)
    df['is_libur'] = ((df['Tanggal'].dt.dayofweek == 6) | (df['is_holiday'] == 1)).astype(int)
    df['MA_7_hari'] = df['Jumlah Terjual'].rolling(window=7).mean(); df['lag_1'] = df['Jumlah Terjual'].shift(1); df['lag_7'] = df['Jumlah Terjual'].shift(7)
    df.fillna(0, inplace=True)
    return df

def load_assets():
    try:
        model = tf.keras.models.load_model('model_ibuprofen_alstm_final.h5', custom_objects={'ALSTMCell': ALSTMCell})
        scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))
        scaler_y = pickle.load(open('scaler_y.pkl', 'rb'))
        all_drug_data = {}
        for drug_id, info in obat_info.items():
            all_drug_data[drug_id] = engineer_features(pd.read_excel(info['file']))
        print(">>> SUKSES: Semua file berhasil dimuat.")
        return model, scaler_X, scaler_y, all_drug_data
    except Exception as e:
        print(f"!!! GAGAL MEMUAT FILE: {e}")
        return None, None, None, {}

model, scaler_X, scaler_y, all_drug_data = load_assets()

@app.route('/')
def index():
    if not all_drug_data:
        return "Gagal memuat data obat.", 500
    return render_template('index.html', drugs=list(obat_info.values()))


@app.route('/predict/<drug_id>')
def predict(drug_id):
    if not all_drug_data: return "Data obat belum dimuat.", 500
    drug_name = obat_info.get(drug_id, {}).get('name')
    if not drug_name: return "Obat tidak ditemukan.", 404
    
    try:
        df = all_drug_data[drug_id]
        
        # Mengambil tanggal terakhir dari data historis untuk default date range picker
        last_historical_date = df['Tanggal'].max()
        start_default_date = last_historical_date - timedelta(days=29)
        
        default_date_range = {
            'start': start_default_date.strftime('%Y-%m-%d'),
            'end': last_historical_date.strftime('%Y-%m-%d')
        }

        feature_columns = scaler_X.feature_names_in_
        time_step = 60
        df_monthly = df.copy()
        df_monthly['Bulan'] = df_monthly['Tanggal'].dt.to_period('M')
        monthly_sales = df_monthly.groupby('Bulan')['Jumlah Terjual'].sum()
        monthly_chart_data = {
            "labels": [period.strftime('%b %Y') for period in monthly_sales.index],
            "data": [int(value) for value in monthly_sales.values]
        }
        X_scaled = scaler_X.transform(df[feature_columns])
        y_scaled = scaler_y.transform(df[['Jumlah Terjual']])
        def create_sequences(X, y, time_steps=60):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X[i:(i + time_steps)])
                ys.append(y[i + time_steps])
            return np.array(Xs), np.array(ys)
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_step)
        train_size_split = int(len(X_seq) * 0.8)
        X_test, y_test = X_seq[train_size_split:], y_seq[train_size_split:]
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred_perf = scaler_y.inverse_transform(y_pred_scaled)
        y_test_actual = scaler_y.inverse_transform(y_test)
        num_test_samples = len(y_test_actual)
        train_plot_df = df.iloc[:-num_test_samples]
        test_plot_df = df.iloc[-num_test_samples:]
        
        df['Trend_1_Tahun'] = df['Jumlah Terjual'].rolling(window=365).mean()
        trend_plot_data = df.dropna(subset=['Trend_1_Tahun'])

        performance_chart_data = {
            "train_labels": [d.strftime('%Y-%m-%d') for d in train_plot_df['Tanggal']],
            "train_data": [int(v) for v in train_plot_df['Jumlah Terjual']],
            "actual_labels": [d.strftime('%Y-%m-%d') for d in test_plot_df['Tanggal']],
            "actual_data": [int(v) for v in y_test_actual.flatten()],
            "predicted_data": [int(v) for v in y_pred_perf.flatten()],
            "trend_labels": [d.strftime('%Y-%m-%d') for d in trend_plot_data['Tanggal']],
            "trend_data": [v for v in trend_plot_data['Trend_1_Tahun']]
        }

        last_date = df['Tanggal'].iloc[-1]
        prediction_start_date = (last_date + pd.offsets.MonthBegin(1))
        prediction_period = prediction_start_date.strftime('%B %Y')
        gap_days = (prediction_start_date - last_date).days
        current_sequence = np.array([scaler_X.transform(df[feature_columns].tail(time_step))])
        last_sales = list(df['Jumlah Terjual'].tail(6))
        pred_years = set(pd.date_range(start=last_date, periods=31 + gap_days, freq='D').year)
        indonesia_holidays = holidays.Indonesia(years=list(pred_years))
        holiday_dates = set(indonesia_holidays.keys())
        one_scaled = scaler_y.transform(np.array([[1]]))
        if gap_days > 0:
            temp_date = last_date
            for _ in range(gap_days):
                last_sales.append(0.0)
                temp_date += timedelta(days=1)
                is_libur = (temp_date.weekday() == 6) or (temp_date.date() in holiday_dates)
                new_features = {'Bulan': temp_date.month, 'Hari': temp_date.day, 'Hari_Ke': temp_date.dayofyear, 'is_libur': 1 if is_libur else 0, 'is_holiday': 1 if temp_date.date() in holiday_dates else 0, 'day_of_week': temp_date.dayofweek, 'is_weekend': 1 if temp_date.dayofweek >= 5 else 0, 'is_start_of_month': 1 if temp_date.is_month_start else 0, 'is_end_of_month': 1 if temp_date.is_month_end else 0, 'week_of_year': temp_date.isocalendar()[1], 'quarter': temp_date.quarter, 'MA_7_hari': np.mean(last_sales[-7:]), 'lag_1': last_sales[-2], 'lag_7': last_sales[-8] if len(last_sales) > 7 else 0}
                new_features_df = pd.DataFrame([new_features], columns=feature_columns)
                new_features_scaled = scaler_X.transform(new_features_df)
                current_sequence = np.append(current_sequence[:, 1:, :], new_features_scaled.reshape(1, 1, len(feature_columns)), axis=1)
        future_predictions_scaled = []
        for i in range(30):
            pred_scaled = model.predict(current_sequence, verbose=0)
            next_date = prediction_start_date + timedelta(days=i)
            is_libur = (next_date.weekday() == 6) or (next_date.date() in holiday_dates)
            if is_libur:
                pred_scaled[0, 0] = one_scaled[0, 0]
            future_predictions_scaled.append(pred_scaled[0, 0])
            pred_actual = scaler_y.inverse_transform(pred_scaled)[0, 0]
            last_sales.append(pred_actual)
            next_iter_date = next_date + timedelta(days=1)
            is_next_iter_libur = (next_iter_date.weekday() == 6) or (next_iter_date.date() in holiday_dates)
            new_features = {'Bulan': next_iter_date.month, 'Hari': next_iter_date.day, 'Hari_Ke': next_iter_date.dayofyear,'is_libur': 1 if is_next_iter_libur else 0, 'is_holiday': 1 if next_iter_date.date() in holiday_dates else 0,'day_of_week': next_iter_date.dayofweek, 'is_weekend': 1 if next_iter_date.dayofweek >= 5 else 0,'is_start_of_month': 1 if next_iter_date.is_month_start else 0, 'is_end_of_month': 1 if next_iter_date.is_month_end else 0,'week_of_year': next_iter_date.isocalendar()[1], 'quarter': next_iter_date.quarter,'MA_7_hari': np.mean(last_sales[-7:]),'lag_1': last_sales[-2], 'lag_7': last_sales[-8] if len(last_sales) > 7 else 0}
            new_features_df = pd.DataFrame([new_features], columns=feature_columns)
            new_features_scaled = scaler_X.transform(new_features_df)
            current_sequence = np.append(current_sequence[:, 1:, :], new_features_scaled.reshape(1, 1, len(feature_columns)), axis=1)
        predictions = scaler_y.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
        predictions = np.round(predictions.flatten()).astype(int)
        
        future_dates = pd.date_range(start=prediction_start_date, periods=30, freq='D')
        
        historical_comparison_data = []
        for i in range(1, 4):
            start_date_past = prediction_start_date - pd.DateOffset(years=i)
            end_date_past = future_dates[-1] - pd.DateOffset(years=i)
            
            past_year_df = df[(df['Tanggal'] >= start_date_past) & (df['Tanggal'] <= end_date_past)]
            
            full_dates_past = pd.date_range(start=start_date_past, periods=30, freq='D')
            past_year_sales = past_year_df.set_index('Tanggal')['Jumlah Terjual'].reindex(full_dates_past).fillna(0).values
            
            historical_comparison_data.append({
                'label': str(start_date_past.year),
                'data': [int(v) for v in past_year_sales]
            })

        forecast_chart_data = {
            "labels": [d.strftime('%d %b') for d in future_dates],
            "data": [int(p) for p in predictions],
            "historical_data": historical_comparison_data
        }

        total_prediction = int(np.sum(predictions))
        prediction_details = [
            {'tanggal': d.strftime('%d %B %Y'), 'prediksi': int(p)}
            for d, p in zip(future_dates, predictions)
        ]
        is_main_model = True if drug_id == 'ibuprofen' else False

        return render_template('predict.html', 
                               drug_id=drug_id,
                               drug_name=drug_name, 
                               is_main_model=is_main_model,
                               monthly_chart_data=monthly_chart_data,
                               performance_chart_data=performance_chart_data,
                               forecast_chart_data=forecast_chart_data,
                               total_prediction=total_prediction,
                               prediction_details=prediction_details,
                               default_date_range=default_date_range, # Mengirim rentang tanggal default
                               prediction_period=prediction_period)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Terjadi kesalahan: {e}", 500

# Endpoint lama, bisa dihapus jika sudah tidak dipakai
@app.route('/historical_data/<drug_id>/<year_month>')
def historical_data(drug_id, year_month):
    # ... (logika endpoint lama tidak perlu diubah, tapi tidak akan dipakai lagi)
    return jsonify({'error': 'Endpoint ini tidak digunakan lagi'}), 404

# <<< ENDPOINT BARU UNTUK RENTANG TANGGAL >>>
@app.route('/historical_range_data/<drug_id>')
def historical_range_data(drug_id):
    if not all_drug_data:
        return jsonify({'error': 'Data belum dimuat'}), 500
    
    df = all_drug_data.get(drug_id)
    if df is None:
        return jsonify({'error': 'Obat tidak ditemukan'}), 404
    
    start_date_str = request.args.get('start')
    end_date_str = request.args.get('end')
    
    if not start_date_str or not end_date_str:
        return jsonify({'error': 'Parameter start dan end diperlukan'}), 400
        
    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
    except ValueError:
        return jsonify({'error': 'Format tanggal tidak valid'}), 400

    range_data = df[(df['Tanggal'] >= start_date) & (df['Tanggal'] <= end_date)]
    
    if range_data.empty:
        return jsonify({'labels': [], 'data': []})
        
    chart_data = {
        'labels': [d.strftime('%d %b') for d in range_data['Tanggal']],
        'data': [int(v) for v in range_data['Jumlah Terjual']]
    }
    return jsonify(chart_data)

if __name__ == '__main__':
    app.run(debug=True)