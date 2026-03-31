import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from forecasting import ForecastModels, calculate_mape, calculate_wape, get_error_breakdown, calculate_mae, calculate_mse, calculate_rmse
from database import FirebaseManager
import google.generativeai as genai

st.set_page_config(page_title="Drug Forecast System", layout="wide")

# Initialize Firebase
db_manager = FirebaseManager()

# --- Sanity Check Functions ---
def validate_data(data_string):
    try:
        df = pd.read_csv(io.StringIO(data_string))
        if 'Demand' not in df.columns:
            return None, "Error: ไม่พบคอลัมน์ 'Demand'"
        return df, "Success"
    except Exception as e:
        return None, f"Error: รูปแบบข้อมูลไม่ถูกต้อง ({str(e)})"

# --- UI Sidebar ---
menu = st.sidebar.selectbox("เมนูการใช้งาน", ["Admin - จัดการข้อมูล", "User - พยากรณ์รายเดือน"])

st.sidebar.divider()
st.sidebar.header("📖 ความรู้เกี่ยวกับโมเดล")

with st.sidebar.expander("1. Naive (โมเดลพื้นฐาน)"):
    st.write("**หลักการ:** ใช้ค่าล่าสุดของเดือนนี้เป็นคำตอบของเดือนหน้าทันที")
    st.latex(r"F_{t+1} = Y_t")
    st.write("**เหมาะสำหรับ:** ข้อมูลที่ไม่มีรูปแบบชัดเจนหรือเปลี่ยนแปลงน้อยมาก")

with st.sidebar.expander("2. Moving Average (MA)"):
    st.write("**หลักการ:** หาค่าเฉลี่ยย้อนหลังตามจำนวนเดือน (Window) ที่กำหนด")
    st.latex(r"F_{t+1} = \frac{1}{n} \sum_{i=0}^{n-1} Y_{t-i}")
    st.write("**เหมาะสำหรับ:** ข้อมูลที่มีการแกว่งตัว แต่ต้องการดูค่ากลางในระยะสั้น")

with st.sidebar.expander("3. Weighted Moving Average (WMA)"):
    st.write("**หลักการ:** คล้าย MA แต่ให้น้ำหนัก (Weight) แต่ละเดือนไม่เท่ากัน")
    st.latex(r"F_{t+1} = \sum_{i=1}^{n} w_i Y_{t-i+1}")
    st.write("**เหมาะสำหรับ:** เมื่อเราเชื่อว่าเดือนที่ใกล้ที่สุดมีผลต่ออนาคตมากกว่าเดือนที่ไกลออกไป")

with st.sidebar.expander("4. Linear Regression (LR)"):
    st.write("**หลักการ:** สร้างเส้นตรงเพื่อหาแนวโน้ม (Trend) ของข้อมูล")
    st.latex(r"y = mx + c")
    st.write("- $m$ คือ ความชัน (Trend)")
    st.write("- $c$ คือ จุดตัดแกน")
    st.write("**เหมาะสำหรับ:** ข้อมูลที่มีแนวโน้มเพิ่มขึ้นหรือลดลงอย่างชัดเจน")

with st.sidebar.expander("5. ARIMA"):
    st.write("**หลักการ:** โมเดลสถิติขั้นสูงที่ผสมผสาน 3 ส่วน:")
    st.write("- **AR (p):** การนำค่าในอดีตมาพยากรณ์ตัวมันเอง")
    st.write("- **I (d):** การหาผลต่างเพื่อให้ข้อมูลนิ่ง (Stationary)")
    st.write("- **MA (q):** การนำค่าความคลาดเคลื่อนในอดีตมาปรับปรุง")
    st.write("**สัมประสิทธิ์ (Coefficients):**")
    st.write("- **หาไปทำไม:** เพื่อหาน้ำหนักที่ 'เหมาะสมที่สุด' ของข้อมูลในอดีตและความผิดพลาดสะสม ที่ทำให้โมเดล 'เรียนรู้' พฤติกรรมการเบิกยาได้แม่นยำที่สุด")
    st.write("- **เอาไปใช้งานยังไง:** ระบบจะใช้ค่าเหล่านี้เป็นตัวคูณกับยอดการใช้ยาในอดีตและค่าความคลาดเคลื่อน เพื่อคำนวณออกมาเป็นตัวเลขพยากรณ์ในเดือนถัดไป")
    st.write("**เหมาะสำหรับ:** ข้อมูลที่มีความซับซ้อน มีรูปแบบความสัมพันธ์ของเวลาชัดเจน")

with st.sidebar.expander("📖 พจนานุกรมศัพท์พยากรณ์"):
    st.write("**- MAPE:** เปอร์เซ็นต์ความผิดพลาดเฉลี่ย ยิ่งน้อยยิ่งแม่นยำ (ควรน้อยกว่า 15-20%)")
    st.write("**- Stationary:** ข้อมูลที่นิ่ง คือข้อมูลที่มีค่าเฉลี่ยและส่วนเบี่ยงเบนคงที่ตลอดเวลา")
    st.write("**- p-value:** ค่าทางสถิติเพื่อตัดสินใจ ถ้า < 0.05 ใน ADF Test แสดงว่าข้อมูลนิ่ง")
    st.write("**- Residual:** ค่าจริง ลบด้วย ค่าพยากรณ์ (สิ่งที่โมเดลยังทำนายไม่ได้)")
    st.write("**- Autocorrelation:** ความสัมพันธ์ของข้อมูลปัจจุบันกับข้อมูลตัวเองในอดีต")

# --- AI Chatbot in Sidebar ---
st.sidebar.divider()
st.sidebar.header("💬 AI Assistant")
user_api_key = st.sidebar.text_input("ใส่ Google API Key เพื่อใช้งาน", type="password", help="รับ API Key ได้ที่ Google AI Studio")

if user_api_key:
    try:
        genai.configure(api_key=user_api_key)

        # --- Health Check: ระบบเลือกโมเดลอัตโนมัติ (Automatic Model Selection) ---
        # ดึงรายชื่อโมเดลที่ API Key นี้สามารถใช้งานได้และรองรับการสร้างเนื้อหา
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # ค้นหาโมเดลตระกูล Gemini ที่เป็นรุ่น 'flash' ซึ่งเหมาะสำหรับงาน Chatbot ที่ต้องการความรวดเร็ว
        model_id = next((m for m in available_models if 'gemini' in m and 'flash' in m), 
                        next((m for m in available_models if 'gemini' in m), 'gemini-1.5-flash'))
        model = genai.GenerativeModel(model_id)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # แสดงประวัติการสนทนาใน Sidebar (ใช้ขนาดกะทัดรัด)
        with st.sidebar.expander("เปิดหน้าต่างแชท", expanded=len(st.session_state.messages) > 0):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            prompt = st.chat_input("ถาม AI ที่นี่...")
            
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    system_context = """
                    คุณคือผู้เชี่ยวชาญด้านการพยากรณ์สต็อกยา (Forecasting Expert) 
                    ระบบนี้ใช้ 5 โมเดล: Naive, Moving Average, WMA, Linear Regression, และ ARIMA
                    หน้าที่ของคุณคืออธิบายหลักการ พารามิเตอร์ และให้คำแนะนำจากผลลัพธ์
                    **คำสั่งสำคัญ: ตอบเป็นภาษาไทยให้กระชับ ตรงประเด็นที่สุด และหลีกเลี่ยงน้ำเยอะ**
                    """
                    full_prompt = f"{system_context}\nคำถาม: {prompt}"
                    
                    with st.spinner("AI กำลังคิด..."):
                        response = model.generate_content(full_prompt)
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                        
        if st.sidebar.button("ล้างการสนทนา"):
            st.session_state.messages = []
            st.rerun()
            
    except Exception as e:
        st.sidebar.error(f"API Key ไม่ถูกต้องหรือเกิดข้อผิดพลาด: {e}")
else:
    st.sidebar.info("💡 กรุณาใส่ API Key ใน Sidebar เพื่อเปิดใช้งานผู้ช่วย AI")

if menu == "Admin - จัดการข้อมูล":
    st.title("Admin Dashboard - การพยาการณ์ความต้องการยาของศูนย์แพทย์")

    # Initialize last_eval in session_state to prevent AttributeError during the Sanity Check below
    if "last_eval" not in st.session_state:
        st.session_state.last_eval = None

    # --- ส่วนที่ 1: การจัดการข้อมูลยา (Data Management) ---
    st.header("📦 ส่วนที่ 1: การจัดการข้อมูลยา (Data Management)")

    # ดึงรายชื่อยาจากฐานข้อมูล
    drug_list = db_manager.get_all_drug_names()

    if not drug_list:
        st.warning("⚠️ ยังไม่มีรายชื่อยาในฐานข้อมูล กรุณาระบุชื่อยาเพื่อเริ่มต้น")
        selected_drug = st.text_input("ชื่อยาที่ต้องการเพิ่ม (เช่น amLODIPine)")
        current_demand = []
    else:
        col_select, col_new = st.columns([2, 1])
        with col_select:
            selected_drug = st.selectbox("เลือกชนิดยาเพื่อจัดการ", drug_list)
        with col_new:
            if st.checkbox("➕ เพิ่มยาชนิดใหม่"):
                selected_drug = st.text_input("ระบุชื่อยาใหม่")
                current_demand = []
            else:
                current_demand = db_manager.get_drug_data(selected_drug)

    if selected_drug:
        # Sanity Check: ถ้าเปลี่ยนชนิดยา ให้ล้างผลการประเมินเก่าออก
        if st.session_state.last_eval is not None and st.session_state.last_eval.get("drug_name") != selected_drug:
            st.session_state.last_eval = None

        st.info(f"ข้อมูลปัจจุบันของ {selected_drug}: {len(current_demand)} จุด")
        
        # กำหนดค่าเริ่มต้นสำหรับ test_size
        test_size = 6

        # แสดงกราฟข้อมูล
        if current_demand:
            # เพิ่มการ Input จำนวน Test Set เองในส่วนจัดการข้อมูล
            max_test = max(1, len(current_demand) - 1)
            test_size = st.number_input("กำหนดจำนวนเดือนสำหรับ Test Set (Evaluation Window)", 
                                        min_value=1, max_value=max_test, value=min(6, max_test))

            # --- EDA Section ---
            st.subheader("📊 ข้อมูลเชิงสถิติ (EDA)")
            eda_res = ForecastModels.run_eda(pd.Series(current_demand))
            
            if eda_res:
                col_eda1, col_eda2, col_eda3, col_eda4 = st.columns(4)
                col_eda1.metric("ความผันผวน (CV)", f"{eda_res['CV (%)']:.2f}%", help="สัมประสิทธิ์ความแปรผัน: บอกถึงความผันผวนของข้อมูลเทียบกับค่าเฉลี่ย หากค่าสูง (>30%) แสดงว่าข้อมูลมีความผันผวนสูงและอาจพยากรณ์ได้ยาก")
                col_eda2.metric("ความนิ่ง (Stationary)", eda_res['Stationary'], help=f"การทดสอบความนิ่ง (ADF Test): p-value = {eda_res['ADF p-value']:.4f}. หากผลเป็น 'Yes' (p < 0.05) แสดงว่าข้อมูลมีคุณสมบัติทางสถิตินิ่ง ซึ่งเหมาะมากสำหรับการใช้โมเดล ARIMA")
                col_eda3.metric("แนวโน้ม (Trend)", f"{eda_res['Trend Slope']:.2f}", help="ค่าความชันของแนวโน้ม: บอกทิศทางของข้อมูลโดยรวม ค่าบวกหมายถึงแนวโน้มความต้องการเพิ่มขึ้น ค่าลบหมายถึงแนวโน้มลดลง")
                col_eda4.metric("Max/Min Ratio", f"{eda_res['Max/Min Ratio']:.2f}x", help="อัตราส่วนค่าสูงสุดต่อค่าต่ำสุด: ใช้ดูความกว้างของการแกว่งตัวของข้อมูลในชุดนี้ ยิ่งค่าสูงแสดงว่าช่วงการเบิกยามีความแตกต่างกันมากในแต่ละเดือน")

                if eda_res.get('Zero Proportion', 0) > 0.3:
                    st.warning(f"⚠️ ยานี้มียอดเป็น 0 ถึง {eda_res['Zero Proportion']*100:.0f}% โมเดล ARIMA อาจทำงานได้ไม่ดี แนะนำให้ใช้ Moving Average หรือ Naive แทน")

                with st.expander("🔍 วิเคราะห์ความสัมพันธ์ย้อนหลัง (ACF/PACF) สำหรับ ARIMA"):
                    # คำนวณเส้นนัยสำคัญ (95% Confidence Interval)
                    conf_interval = 1.96 / (len(current_demand)**0.5)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_acf, ax_acf = plt.subplots(figsize=(5, 3))
                        ax_acf.bar(range(len(eda_res['ACF'])), eda_res['ACF'], color='skyblue')
                        ax_acf.axhline(y=conf_interval, linestyle='--', color='red', alpha=0.5)
                        ax_acf.axhline(y=-conf_interval, linestyle='--', color='red', alpha=0.5)
                        ax_acf.axhline(y=0, color='black', linewidth=0.8)
                        ax_acf.set_title("ACF (q)")
                        st.pyplot(fig_acf)
                        st.caption(f"เส้นประสีแดงคือนัยสำคัญ (±{conf_interval:.2f})")
                    with c2:
                        fig_pacf, ax_pacf = plt.subplots(figsize=(5, 3))
                        ax_pacf.bar(range(len(eda_res['PACF'])), eda_res['PACF'], color='salmon')
                        ax_pacf.axhline(y=conf_interval, linestyle='--', color='red', alpha=0.5)
                        ax_pacf.axhline(y=-conf_interval, linestyle='--', color='red', alpha=0.5)
                        ax_pacf.axhline(y=0, color='black', linewidth=0.8)
                        ax_pacf.set_title("PACF (p)")
                        st.pyplot(fig_pacf)
                        st.caption("แท่งที่สูงทะลุเส้นประสีแดงคือค่าที่ควรนำมาตั้งค่า p")

            col_chart1, col_chart2 = st.columns([3, 1])
            with col_chart1:
                fig, ax = plt.subplots(figsize=(10, 4))
                n = len(current_demand)
                if n > test_size:
                    train_idx = n - test_size
                    ax.plot(range(train_idx + 1), current_demand[:train_idx + 1], color='#1f77b4', label='Train Data', marker='o', markersize=3)
                    ax.plot(range(train_idx, n), current_demand[train_idx:], color='#ff7f0e', label=f'Test Data', linestyle='--', marker='o', markersize=3)
                    ax.axvline(x=train_idx, color='red', linestyle=':', alpha=0.7)
                else:
                    ax.plot(current_demand, marker='o', color='#1f77b4', label='Demand Data')
                ax.set_ylabel("Demand Units")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col_chart2:
                fig_box, ax_box = plt.subplots(figsize=(4, 6))
                ax_box.boxplot(current_demand)
                ax_box.set_title("Outliers Check")
                ax_box.set_xticklabels([selected_drug])
                ax_box.grid(True, alpha=0.2)
                st.pyplot(fig_box)

        # เครื่องมือจัดการข้อมูล
        col_edit, col_del = st.columns(2)
        
        with col_edit:
            with st.expander("📝 เพิ่มหรือแก้ไขข้อมูล (Add/Edit Data)"):
                # สร้าง csv string จากข้อมูลปัจจุบันเพื่อให้ง่ายต่อการแก้ไข
                current_csv = "Demand\n" + "\n".join(map(str, current_demand)) if current_demand else "Demand"
                raw_input = st.text_area("วางข้อมูล CSV (ระบบจะบันทึกทับข้อมูลเดิมหรือสร้างใหม่)", value=current_csv, height=150)
                if st.button("บันทึกข้อมูล (Save to DB)"):
                    df, msg = validate_data(raw_input)
                    if df is not None:
                        new_vals = df['Demand'].tolist()
                        if db_manager.save_drug_data(selected_drug, new_vals):
                            st.success(f"บันทึกข้อมูล {selected_drug} เรียบร้อยแล้ว")
                            st.rerun()
                    else:
                        st.error(msg)

        with col_del:
            if drug_list and selected_drug in drug_list:
                with st.expander("🗑️ ลบข้อมูลยา (Delete)"):
                    st.warning(f"การลบยา '{selected_drug}' จะไม่สามารถกู้คืนได้")
                    confirm_delete = st.checkbox("ฉันยืนยันว่าต้องการลบ")
                    if st.button("ยืนยันการลบ", type="primary", disabled=not confirm_delete):
                        if db_manager.delete_drug_data(selected_drug):
                            st.success("ลบข้อมูลสำเร็จ")
                            st.rerun()

    st.divider()

    # --- ส่วนที่ 2: ตั้งค่าพารามิเตอร์โมเดล (Parameter Settings) ---
    st.header("⚙️ ส่วนที่ 2: ตั้งค่าพารามิเตอร์โมเดล (Parameter Settings)")
    
    tabs = st.tabs(["ARIMA", "MA & WMA", "Linear Regression"])

    with tabs[0]:
        st.subheader("ARIMA Configuration")
        col_p, col_d, col_q = st.columns(3)
        with col_p:
            p = st.number_input("p", min_value=0, max_value=12, value=6)
        with col_d:
            d = st.number_input("d", min_value=0, max_value=1, value=1)
        with col_q:
            q = st.number_input("q", min_value=0, max_value=12, value=1)

    with tabs[1]:
        st.subheader("MA & WMA Settings")
        col_ma, col_wma = st.columns(2)
        with col_ma:
            ma_window = st.number_input("MA Window (Period)", min_value=1, max_value=12, value=6)
        with col_wma:
            wma_window = st.number_input("WMA Window (Period)", min_value=1, max_value=12, value=6)
        
        st.write(f"WMA Weights (รวมต้องได้ 1.0)")
        cols_w = st.columns(3)
        wma_weights = []
        default_w = round(1.0 / wma_window, 2)
        for i in range(wma_window):
            # ปรับมุมมองใหม่: t คือเดือนที่กำลังจะทำนาย
            # ข้อมูลล่าสุดที่มีคือ t-1, เก่าลงไปคือ t-2, ...
            dist = wma_window - i
            label = f"น้ำหนักข้อมูล: {dist} เดือนก่อนหน้า (t-{dist})"
            with cols_w[i % 3]:
                w = st.number_input(label, min_value=0.0, max_value=1.0, value=default_w, step=0.01, format="%.2f", key=f"wma_idx_{i}")
                wma_weights.append(w)
        
        current_sum = sum(wma_weights)
        if abs(current_sum - 1.0) > 0.01:
            st.warning(f"⚠️ ผลรวมน้ำหนัก: {current_sum:.2f} (ควรเป็น 1.0)")

    with tabs[2]:
        st.subheader("Linear Regression Configuration")
        lr_window = st.number_input("Linear Regression Window (0 = ทั้งหมด)", min_value=0, max_value=100, value=0)

    st.divider()

    # --- ส่วนที่ 3: ประเมินผลโมเดล (Model Evaluation) ---
    st.header("🚀 ส่วนที่ 3: ประเมินผลโมเดล (Model Evaluation)")
    
    if st.button("Run Model Evaluation", type="primary"):
        if not current_demand:
            st.error("กรุณาเลือกยาและเพิ่มข้อมูลก่อนทำการประเมินผล")
        elif len(current_demand) <= test_size:
            st.warning(f"ข้อมูลไม่พอสำหรับการแบ่ง Train/Test (ต้องการอย่างน้อย {test_size + 1} จุด, ปัจจุบันมี {len(current_demand)})")
        else:
            with st.spinner("Evaluating models..."):
                actuals = current_demand[-test_size:]
                arima_label = f"ARIMA ({p},{d},{q})"
                ma_label = "Moving Average"
                wma_label = "Weighted Moving Average"
                lr_label = "Linear Regression"
                model_names = ["Naive", ma_label, wma_label, lr_label, arima_label]
                all_preds = {name: [] for name in model_names}
                
                # Walk-forward Validation
                for i in range(len(current_demand) - test_size, len(current_demand)):
                    history = pd.Series(current_demand[:i])
                    
                    all_preds["Naive"].append(ForecastModels.naive(history))
                    all_preds[ma_label].append(ForecastModels.moving_average(history, window=ma_window))
                    all_preds[wma_label].append(ForecastModels.weighted_moving_average(history, weights=wma_weights))
                    all_preds[lr_label].append(ForecastModels.linear_regression(history, window=lr_window if lr_window > 0 else None))
                    all_preds[arima_label].append(ForecastModels.arima(history, order=(p, d, q)))

                # คำนวณ MAPE
                mape_results = []
                min_mape = float('inf')
                best_model_name = ""
                
                for name in model_names:
                    mape_val = calculate_mape(actuals, all_preds[name])
                    wape_val = calculate_wape(actuals, all_preds[name])
                    mae_val = calculate_mae(actuals, all_preds[name])
                    mse_val = calculate_mse(actuals, all_preds[name])
                    rmse_val = calculate_rmse(actuals, all_preds[name])
                    
                    mape_results.append({
                        "Model": name, 
                        "MAPE (%)": f"{mape_val:.2f}%",
                        "WAPE (%)": f"{wape_val:.2f}%",
                        "MAE": f"{mae_val:.2f}",
                        "MSE": f"{mse_val:.2f}",
                        "RMSE (Units)": f"{rmse_val:.2f}"
                    })
                    if wape_val < min_mape:
                        min_mape = wape_val
                        best_model_name = name
                
                # คำนวณ Insights เพียงครั้งเดียวที่นี่เพื่อประสิทธิภาพ
                arima_ins = ForecastModels.get_model_insights(pd.Series(current_demand), "ARIMA", order=(p, d, q))
                lr_ins = ForecastModels.get_model_insights(pd.Series(current_demand), "Linear Regression", window=lr_window if lr_window > 0 else None)
                
                # เตรียมข้อมูล Residual สำหรับทุกโมเดล
                residual_dict = {}
                for name in model_names:
                    preds = all_preds[name]
                    residual_dict[name] = [a - f for a, f in zip(actuals, preds)]
                
                # เก็บผลลัพธ์ไว้ใน session_state
                st.session_state.last_eval = {
                    "drug_name": selected_drug,
                    "mape_results": mape_results,
                    "all_preds": all_preds,
                    "actuals": actuals,
                    "residual_dict": residual_dict,
                    "model_names": model_names,
                    "arima_label": arima_label,
                    "best_model_name": best_model_name,
                    "min_mape": min_mape,
                    "p": p, "d": d, "q": q,
                    "wma_weights": wma_weights,
                    "arima_ins": arima_ins,
                    "lr_ins": lr_ins
                }
                
                db_manager.save_model_config(selected_drug, {
                    "best_model": best_model_name,
                    "arima_order": [p, d, q],
                    "ma_window": ma_window,
                    "lr_window": lr_window,
                    "wma_weights": wma_weights,
                    "mape": min_mape
                })

    # แสดงผลลัพธ์จาก Session State (ถ้ามี)
    if st.session_state.last_eval:
        eval_data = st.session_state.last_eval
        mape_results = eval_data["mape_results"]
        all_preds = eval_data["all_preds"]
        actuals = eval_data["actuals"]
        model_names = eval_data["model_names"]
        best_model_name = eval_data["best_model_name"]
        min_mape = eval_data["min_mape"]
        residual_dict = eval_data["residual_dict"]
        arima_label = eval_data["arima_label"]
        p, d, q = eval_data["p"], eval_data["d"], eval_data["q"]
        wma_weights = eval_data["wma_weights"]
        arima_ins = eval_data["arima_ins"]
        lr_ins = eval_data["lr_ins"]

        # ส่วนการแสดงผลเดิม
        st.subheader("📊 ผลการประเมินโมเดล")
        
        st.write("### 📏 ตารางเปรียบเทียบค่าความคลาดเคลื่อน (MAPE)")
        st.table(pd.DataFrame(mape_results))

        st.divider()

        st.write("### 📈 กราฟเปรียบเทียบค่าจริงกับค่าพยากรณ์ (Actual vs Predictions)")
        
        # --- กราฟแสดงทุกโมเดลพร้อมกัน (Overall Comparison) ---
        fig_all, ax_all = plt.subplots(figsize=(10, 5))
        months_label = [f"M+{j+1}" for j in range(len(actuals))]
        # Plot Actual เป็นเส้นหนาสีดำให้อยู่บนสุด (zorder=5)
        ax_all.plot(months_label, actuals, label='Actual (ค่าจริง)', color='black', linewidth=3, marker='s', zorder=5)
        
        # Plot ทุกโมเดลวนตาม List สีอัตโนมัติ
        for name in model_names:
            ax_all.plot(months_label, all_preds[name], label=name, marker='o', markersize=4, alpha=0.8)
            
        ax_all.set_title("Overall Comparison: All Models vs Actual")
        ax_all.set_ylabel("Demand Units")
        ax_all.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # วาง Legend ไว้นอกกราฟเพื่อไม่ให้บังเส้น
        ax_all.grid(True, alpha=0.3)
        st.pyplot(fig_all)

        for name in model_names:
            fig_eval, ax_eval = plt.subplots(figsize=(10, 4))
            months_label = [f"M+{j+1}" for j in range(len(actuals))]
            ax_eval.plot(months_label, actuals, label='Actual', color='black', linewidth=2, marker='s')
            ax_eval.plot(months_label, all_preds[name], label=f'Predict ({name})', color='#ff7f0e', linestyle='--')
            ax_eval.set_title(f"Evaluation Model: {name}")
            ax_eval.set_ylabel("Demand Units")
            ax_eval.legend()
            ax_eval.grid(True, alpha=0.3)
            st.pyplot(fig_eval)

        st.success(f"✅ บันทึกโมเดลที่ดีที่สุดเรียบร้อย: {best_model_name} (MAPE: {min_mape:.2f}%)")

        # Insights และ Residual Analysis
        st.divider()
        st.subheader("🧠 ข้อมูลเชิงลึกของโมเดล (Model Insights)")
        ins_col1, ins_col2 = st.columns(2)
        with ins_col1:
            st.write(f"**{arima_label} Coefficients**")
            st.json(arima_ins)
            
            if "Error" not in arima_ins:
                with st.expander("💡 อ่านคำอธิบายพารามิเตอร์ ARIMA"):
                    explanations = ForecastModels.interpret_arima(arima_ins)
                    for exp in explanations:
                        st.write(f"- {exp}")
        with ins_col2:
            st.write(f"**Weighted Moving Average Weights**")
            wma_df = pd.DataFrame({"Month Lag": [f"M-{len(wma_weights)-i}" for i in range(len(wma_weights))], "Weight": wma_weights})
            st.bar_chart(wma_df.set_index("Month Lag"))
            
            st.write("**Linear Regression Equation**")
            if "Slope (Trend)" in lr_ins:
                st.latex(f"y = {lr_ins['Slope (Trend)']:.2f}x + {lr_ins['Intercept']:.2f}")
                
                with st.expander("💡 อ่านคำอธิบาย Linear Regression"):
                    lr_exps = ForecastModels.interpret_linear_regression(lr_ins)
                    for exp in lr_exps:
                        st.write(f"- {exp}")

        st.divider()
        st.subheader("🧐 การวิเคราะห์ค่าความคลาดเคลื่อน (Residual Analysis)")
        
        # เพิ่มส่วนตรวจสอบรายละเอียดการคำนวณ
        with st.expander("🔍 ตรวจสอบรายละเอียดการคำนวณ Error (Calculation Audit)"):
            selected_model_audit = st.selectbox("เลือกโมเดลที่ต้องการตรวจสอบ", model_names)
            breakdown_df = get_error_breakdown(actuals, all_preds[selected_model_audit])
            
            # คำนวณค่าสะสมสำหรับแสดงในสูตร
            total_abs_error = breakdown_df['Abs Error'].sum()
            total_sq_error = breakdown_df['Squared Error'].sum()
            total_actual = breakdown_df['Actual'].sum()
            total_pct_error = breakdown_df['% Error (Point)'].sum()
            n = len(breakdown_df)
            
            audit_wape = (total_abs_error / total_actual * 100) if total_actual != 0 else 0
            audit_mape = total_pct_error / n
            audit_mae = total_abs_error / n
            audit_rmse = (total_sq_error / n) ** 0.5

            st.dataframe(breakdown_df.style.format("{:.2f}").background_gradient(subset=["% Error (Point)"], cmap="Reds"))
            
            st.markdown(f"#### 🧮 การแยกคำนวณราย Metrics สำหรับ {selected_model_audit}")
            c_audit1, c_audit2 = st.columns(2)
            with c_audit1:
                st.info(f"**WAPE:** (Σ|Error| / ΣActual) × 100\n\n({total_abs_error:,.2f} / {total_actual:,.2f}) × 100 = **{audit_wape:.2f}%**")
                st.info(f"**MAPE:** (Σ% Error / n)\n\n({total_pct_error:,.2f} / {n}) = **{audit_mape:.2f}%**")
            with c_audit2:
                st.info(f"**MAE:** (Σ|Error| / n)\n\n({total_abs_error:,.2f} / {n}) = **{audit_mae:.2f}**")
                st.info(f"**RMSE:** √(ΣError² / n)\n\n√({total_sq_error:,.2f} / {n}) = **{audit_rmse:.2f}**")

        res_df = pd.DataFrame(residual_dict, index=[f"M+{j+1}" for j in range(len(actuals))])
        st.dataframe(res_df.style.format("{:.2f}"))
        for name in model_names:
            fig_res, ax_res = plt.subplots(figsize=(10, 2.5))
            ax_res.bar(res_df.index, residual_dict[name], color='gray', alpha=0.6)
            ax_res.axhline(y=0, color='red', linestyle='-')
            ax_res.set_title(f"Residuals: {name}")
            st.pyplot(fig_res)

elif menu == "User - พยากรณ์รายเดือน":
    st.header("💊 User: พยากรณ์ยอดสต็อก")
    
    drug_list = db_manager.get_all_drug_names()
    selected_drug = st.selectbox("เลือกชนิดยา", drug_list)
    
    config = db_manager.get_model_config(selected_drug) if selected_drug else None
    available_models = ["Naive", "Moving Average", "WMA", "Linear Regression", "ARIMA"]

    if config:
        st.success(f"ระบบจะใช้โมเดลที่ดีที่สุดสำหรับยานี้: **{config['best_model']}**")
        default_idx = available_models.index(config['best_model']) if config['best_model'] in available_models else 0
    else:
        st.warning("⚠️ ยังไม่มีการประเมินโมเดลสำหรับยานี้ ระบบจะใช้ค่าเริ่มต้น")
        default_idx = 0

    selected_model = st.selectbox("เลือกโมเดลที่ต้องการใช้งาน", available_models, index=default_idx)

    # คำนวณหาจำนวนข้อมูลขั้นต่ำ (required_len) เฉพาะโมเดลที่เลือก
    if selected_model == "Naive":
        required_len = 1
    elif selected_model == "Moving Average":
        required_len = config.get('ma_window', 6) if config else 6
    elif selected_model == "WMA":
        required_len = len(config['wma_weights']) if config else 6
    elif selected_model == "Linear Regression":
        # LR ควรมีข้อมูลอย่างน้อย 3-6 จุดเพื่อให้เห็นแนวโน้ม
        lr_cfg = config.get('lr_window', 0) if config else 0
        required_len = lr_cfg if lr_cfg > 0 else 6
    elif selected_model == "ARIMA":
        p_val = config['arima_order'][0] if config else 6
        required_len = max(p_val, 1)

    st.subheader(f"กรอกข้อมูลย้อนหลัง {required_len} เดือน เพื่อพยากรณ์เดือนถัดไป (t)")
    
    col1, col2, col3 = st.columns(3)
    inputs = []
    for i in range(required_len):
        dist = required_len - i
        with [col1, col2, col3][i % 3]:
            val = st.number_input(f"{dist} เดือนก่อนหน้า (t-{dist})", min_value=0, key=f"user_in_{i}")
            inputs.append(val)
            
    if st.button("ทำนายผลเดือนถัดไป"):
        if any(v == 0 for v in inputs):
            st.warning("⚠️ พบค่าเป็น 0 ในข้อมูลนำเข้า โปรดตรวจสอบความถูกต้อง")
        
        history = pd.Series(inputs)
        
        # ดึงค่าพารามิเตอร์จาก Config
        order = tuple(config['arima_order']) if config else (1,1,1)
        ma_win = config.get('ma_window', 6) if config else 6
        lr_win = config.get('lr_window', 0) if config else 0
        weights = config['wma_weights'] if config else [0.05, 0.1, 0.15, 0.2, 0.25, 0.25]

        # คำนวณเฉพาะโมเดลที่เลือก หรือแสดงผลแยกให้ชัดเจน
        st.divider()
        st.subheader(f"📊 ผลการพยากรณ์ด้วย {selected_model}")
        
        if selected_model == "Naive":
            val = ForecastModels.naive(history)
        elif selected_model == "Moving Average":
            val = ForecastModels.moving_average(history, window=ma_win)
        elif selected_model == "WMA":
            val = ForecastModels.weighted_moving_average(history, weights=weights)
        elif selected_model == "Linear Regression":
            val = ForecastModels.linear_regression(history, window=lr_win if lr_win > 0 else None)
        elif selected_model == "ARIMA":
            val = ForecastModels.arima(history, order=order)
            
        st.metric(f"ปริมาณที่ควรสำรอง (เดือนถัดไป)", f"{val:,.0f} หน่วย")
