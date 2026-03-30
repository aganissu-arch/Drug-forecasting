import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

class ForecastModels:
    @staticmethod
    def naive(history):
        return history.iloc[-1]

    @staticmethod
    def moving_average(history, window=6):
        return history.rolling(window=window).mean().iloc[-1]

    @staticmethod
    def weighted_moving_average(history, weights=[0.05, 0.1, 0.15, 0.2, 0.25, 0.25]):
        # Sanity Check: ตรวจสอบว่าข้อมูลมีเพียงพอกับน้ำหนักหรือไม่
        n = len(weights)
        if len(history) < n:
            return history.mean()
        last_n = history.values[-n:]
        return np.dot(last_n, weights)

    @staticmethod
    def linear_regression(history, window=None):
        # หากกำหนด window ให้ใช้เฉพาะข้อมูลล่าสุดตามจำนวนที่ระบุ
        if window is not None and len(history) > window:
            data = history.iloc[-window:]
        else:
            data = history
            
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        model = LinearRegression().fit(X, y)
        return model.predict([[len(data)]])[0]

    @staticmethod
    def arima(history, order=(6,1,1)):
        try:
            # ใช้ .values เพื่อหลีกเลี่ยงปัญหาเรื่อง Index/Frequency ของ statsmodels
            # enforce_stationarity=False และ enforce_invertibility=False 
            # จะช่วยให้โมเดลสามารถ Fit ข้อมูลที่มีความผันผวนสูงได้ง่ายขึ้น
            model = ARIMA(history.values, order=order, 
                          enforce_stationarity=False, 
                          enforce_invertibility=False)
            # ลดความซับซ้อนของการ Fit เพื่อป้องกันการเกิด Numerical Error
            # หากใช้ maxiter แล้วยังเป็น nan ให้ลองใช้ค่าเริ่มต้น หรือลด Order ลง
            model_fit = model.fit(method_kwargs={'warn_convergence': False})
            return model_fit.forecast(steps=1)[0]
        except Exception:
            return np.nan

    @staticmethod
    def get_model_insights(history, model_type, **kwargs):
        """
        ดึงข้อมูลพารามิเตอร์ภายในของโมเดล (Brain)
        """
        insights = {}
        if model_type == "Linear Regression":
            window = kwargs.get('window')
            data = history.iloc[-window:] if window and len(history) > window else history
            X = np.arange(len(data)).reshape(-1, 1)
            y = data.values
            model = LinearRegression().fit(X, y)
            # แปลง numpy.float64 เป็น python float เพื่อให้ st.json แสดงผลได้
            insights = {
                "Slope (Trend)": float(model.coef_[0]), 
                "Intercept": float(model.intercept_)
            }
            
        elif model_type == "ARIMA":
            order = kwargs.get('order', (1, 1, 1))
            try:
                model = ARIMA(history.values, order=order, 
                              enforce_stationarity=False, enforce_invertibility=False)
                # ทำเช่นเดียวกันในส่วนดึง Insights เพื่อให้ค่าพารามิเตอร์นิ่งที่สุด
                model_fit = model.fit(method_kwargs={'warn_convergence': False})
                # ดึงชื่อพารามิเตอร์และค่ามาจับคู่กัน (เนื่องจาก model_fit.params เป็น numpy array เมื่อส่งข้อมูลแบบ .values)
                params_dict = dict(zip(model_fit.param_names, model_fit.params))
                # แปลงค่าทศนิยมทุกตัวให้เป็น Python float เพื่อให้ st.json แสดงผลได้
                insights = {str(k): float(v) for k, v in params_dict.items()}
            except Exception as e:
                insights = {"Error": f"ไม่สามารถดึงข้อมูลพารามิเตอร์ได้: {str(e)}"}
                
        return insights

    @staticmethod
    def interpret_arima(params_dict):
        """
        สร้างคำอธิบายพารามิเตอร์ ARIMA เป็นภาษาไทย
        """
        explanations = []
        for key, val in params_dict.items():
            if key.startswith('ar.L'):
                lag = key.split('L')[1]
                effect = "ดึงค่ากลับเพื่อรักษาสมดุล (Self-Correction)" if val < 0 else "ส่งเสริมแนวโน้มเดิม (Momentum)"
                explanations.append(f"**{key} (ความจำเดือนที่ {lag})**: มีอิทธิพลแบบ {effect} ด้วยน้ำหนัก {val:.4f}")
            
            elif key.startswith('ma.L'):
                lag = key.split('L')[1]
                effect = "ปรับยอดเพิ่มขึ้นตามความผิดพลาด" if val > 0 else "ลดทอนความผิดพลาดจากเดือนก่อน"
                explanations.append(f"**{key} (การเรียนรู้จากความคลาดเคลื่อน)**: นำความผิดพลาดของเดือนที่ {lag} มา{effect} ด้วยน้ำหนัก {val:.4f}")
            
            elif key == 'sigma2':
                risk_level = "สูง (ข้อมูลมีความผันผวนมาก)" if val > 1000000 else "ปกติ"
                explanations.append(f"**sigma2 (ความเสี่ยง)**: ค่าความแปรปรวนของ Error คือ {val:,.2f} ยิ่งค่านี้สูง หมายความว่าข้อมูลมีความผันผวนหรือความไม่แน่นอน{risk_level}")
            
            elif key == 'const' or key == 'intercept':
                explanations.append(f"**{key} (ค่าคงที่)**: ค่าพื้นฐานของความต้องการยาอยู่ที่ประมาณ {val:.2f} หน่วย")

        # เพิ่มส่วนสรุปวิธีการคำนวณ
        explanations.append("---")
        explanations.append("**💡 เอาไปคำนวณยังไง:**")
        explanations.append("1. **ส่วนต่าง (d=1):** คำนวณหาการเปลี่ยนแปลงจากเดือนก่อนหน้า")
        explanations.append("2. **ส่วน AR:** นำ 'การเปลี่ยนแปลงในอดีต' มาคูณกับน้ำหนัก (เช่น ar.L1) เพื่อดูแรงเหวี่ยง")
        explanations.append("3. **ส่วน MA:** นำ 'ค่าความผิดพลาดเดือนที่แล้ว' มาคูณกับน้ำหนัก (เช่น ma.L1) เพื่อปรับจูน")
        explanations.append("4. **ผลลัพธ์:** นำค่าที่ได้จากข้อ 2 และ 3 มารวมกัน แล้วบวกกลับเข้าไปที่ยอดเดือนล่าสุดเพื่อเป็นผลพยากรณ์")
        
        return explanations

    @staticmethod
    def interpret_linear_regression(ins):
        """
        สร้างคำอธิบายพารามิเตอร์ Linear Regression เป็นภาษาไทย
        """
        slope = ins.get("Slope (Trend)", 0)
        intercept = ins.get("Intercept", 0)
        trend_direction = "เพิ่มขึ้น" if slope > 0 else "ลดลง"
        
        explanations = [
            f"**ความชัน (Slope: {slope:.2f})**: ในทุกๆ 1 เดือนที่ผ่านไป โมเดลคาดการณ์ว่าความต้องการยาจะ **{trend_direction}** เฉลี่ยเดือนละ {abs(slope):.2f} หน่วย",
            f"**ค่าเริ่มต้น (Intercept: {intercept:.2f})**: จุดเริ่มต้นของการคำนวณตามเส้นแนวโน้มอยู่ที่ {intercept:.2f} หน่วย",
            f"**การตีความ**: ข้อมูลชุดนี้มีทิศทางเป็นแนวโน้ม **{trend_direction}** อย่างชัดเจนในระยะยาว"
        ]
        return explanations

    @staticmethod
    def run_eda(series):
        """
        วิเคราะห์ค่าทางสถิติพื้นฐาน (EDA)
        """
        if len(series) < 2:
            return None
            
        # 1. Coefficient of Variation (CV)
        cv = (series.std() / series.mean()) * 100 if series.mean() != 0 else 0
        
        # 2. ADF Test (Stationarity)
        is_stationary = "Unknown"
        p_value = 1.0
        try:
            adf_res = adfuller(series.dropna())
            p_value = adf_res[1]
            is_stationary = "Yes" if p_value < 0.05 else "No"
        except:
            pass
            
        # 3. Trend Slope (Simple)
        # คำนวณหาค่าเฉลี่ยการเปลี่ยนแปลงต่อเดือน (Average Delta)
        # ใช้ n-1 เพื่อหาจำนวนช่วงเวลาที่เกิดขึ้นจริง
        slope = (series.iloc[-1] - series.iloc[0]) / (len(series) - 1)
        
        # 4. Check for Intermittent Demand ( sparsity )
        zero_pct = (series == 0).sum() / len(series)
        
        return {
            "CV (%)": cv,
            "Stationary": is_stationary,
            "ADF p-value": p_value,
            "Trend Slope": slope,
            "Max/Min Ratio": series.max() / series.min() if series.min() != 0 else np.inf,
            "Zero Proportion": zero_pct,
            "ACF": acf(series.dropna(), nlags=min(10, len(series)//2 - 1)).tolist(),
            "PACF": pacf(series.dropna(), nlags=min(10, len(series)//2 - 1)).tolist()
        }

def calculate_wape(actual, forecast):
    """
    คำนวณ WAPE (Weighted Absolute Percentage Error)
    เหมาะสำหรับข้อมูลที่มีค่า 0 เพราะใช้ผลรวมของค่าจริงเป็นตัวหาร
    """
    actual, forecast = np.array(actual, dtype=float), np.array(forecast, dtype=float)
    # กรองเฉพาะจุดที่โมเดลสามารถทำนายได้ (ไม่เป็น NaN)
    mask = ~np.isnan(forecast)
    if not np.any(mask):
        return np.nan
        
    # ในงานพยากรณ์ยา ข้อมูลมักมีค่าเป็น 0 ซึ่งทำให้ MAPE ปกติคำนวณไม่ได้หรือผิดเพี้ยน
    # จึงปรับมาใช้ WAPE (Weighted Absolute Percentage Error) ซึ่งเป็นมาตรฐานสากลในระบบคงคลัง
    # สูตร: (ผลรวมของค่าความคลาดเคลื่อนสัมบูรณ์) / (ผลรวมของค่าจริง)
    total_actual = np.sum(actual[mask])
    total_abs_error = np.sum(np.abs(actual[mask] - forecast[mask]))
    
    if total_actual == 0:
        # หากยอดจริงทั้งหมดเป็น 0 และทำนายเป็น 0 = 0%, หากทำนาย > 0 = 100%
        return 0.0 if total_abs_error == 0 else 100.0
    
    return (total_abs_error / total_actual) * 100

def calculate_mape(actual, forecast):
    """
    คำนวณ Standard MAPE แบบรองรับค่า 0 (Safe MAPE)
    หาค่าเฉลี่ยของ % Error รายเดือน
    """
    actual, forecast = np.array(actual, dtype=float), np.array(forecast, dtype=float)
    mask = ~np.isnan(forecast)
    a, f = actual[mask], forecast[mask]
    if len(a) == 0: return np.nan
    
    pe = []
    for ai, fi in zip(a, f):
        if ai > 0:
            pe.append(abs(ai - fi) / ai)
        else:
            # กรณี Actual=0: หากทาย > 0 ให้ผิด 100%, หากทาย 0 ให้ผิด 0%
            pe.append(1.0 if fi > 0 else 0.0)
            
    return np.mean(pe) * 100

def calculate_mae(actual, forecast):
    actual, forecast = np.array(actual, dtype=float), np.array(forecast, dtype=float)
    mask = ~np.isnan(forecast)
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(actual[mask] - forecast[mask]))

def calculate_mse(actual, forecast):
    actual, forecast = np.array(actual, dtype=float), np.array(forecast, dtype=float)
    mask = ~np.isnan(forecast)
    if not np.any(mask):
        return np.nan
    return np.mean((actual[mask] - forecast[mask])**2)

def calculate_rmse(actual, forecast):
    actual, forecast = np.array(actual, dtype=float), np.array(forecast, dtype=float)
    mask = ~np.isnan(forecast)
    if not np.any(mask):
        return np.nan
    mse = np.mean((actual[mask] - forecast[mask])**2)
    return np.sqrt(mse)

def get_error_breakdown(actual, forecast):
    """
    สร้าง DataFrame แสดงรายละเอียดการคำนวณ Error รายเดือน
    """
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    abs_error = np.abs(actual - forecast)
    sq_error = (actual - forecast)**2
    
    # คำนวณ % Error รายจุดตาม Logic Safe MAPE
    pct_errors = []
    for a, f in zip(actual, forecast):
        if a > 0:
            pct_errors.append((abs(a - f) / a) * 100)
        else:
            # กรณี Actual เป็น 0: หากทายมีค่าให้ผิด 100%, หากทาย 0 ให้ผิด 0%
            pct_errors.append(100.0 if f > 0 else 0.0)
        
    return pd.DataFrame({
        "Actual": actual,
        "Forecast": forecast,
        "Abs Error": abs_error,
        "Squared Error": sq_error,
        "% Error (Point)": pct_errors
    })
