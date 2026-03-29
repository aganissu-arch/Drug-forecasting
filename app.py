import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from forecasting import ForecastModels, calculate_mape
from database import FirebaseManager
import google.generativeai as genai

st.set_page_config(page_title="Drug Forecast System", layout="wide")

# --- Custom CSS for Better UI & Thai Fonts ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    
    div[data-testid="stExpander"] {
        border: none;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        background-color: white;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

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
    st.write("**ขั้นตอนการคำนวณ:**")
    st.write("1. ดูยอดเบิกจริงของเดือนล่าสุด (t-1)")
    st.write("2. นำค่านั้นมาตั้งเป็นยอดพยากรณ์ของเดือนถัดไป (t) ทันที")
    st.write("**เหมาะสำหรับ:** ข้อมูลที่ไม่มีรูปแบบชัดเจนหรือเปลี่ยนแปลงน้อยมาก")

with st.sidebar.expander("2. Moving Average (MA)"):
    st.write("**หลักการ:** หาค่าเฉลี่ยย้อนหลังตามจำนวนเดือน (Window) ที่กำหนด")
    st.latex(r"F_{t+1} = \frac{1}{n} \sum_{i=0}^{n-1} Y_{t-i}")
    st.write("**ขั้นตอนการคำนวณ:**")
    st.write("1. ระบุจำนวนเดือนย้อนหลังที่ต้องการ (เช่น 6 เดือน)")
    st.write("2. รวมยอดเบิกจริงของ 6 เดือนล่าสุดเข้าด้วยกัน")
    st.write("3. หารด้วย 6 เพื่อหาค่าเฉลี่ย และใช้เป็นผลพยากรณ์")
    st.write("**เหมาะสำหรับ:** ข้อมูลที่มีการแกว่งตัว แต่ต้องการดูค่ากลางในระยะสั้น")

with st.sidebar.expander("3. Weighted Moving Average (WMA)"):
    st.write("**หลักการ:** คล้าย MA แต่ให้น้ำหนัก (Weight) แต่ละเดือนไม่เท่ากัน")
    st.latex(r"F_{t+1} = \sum_{i=1}^{n} w_i Y_{t-i+1}")
    st.write("**ขั้นตอนการคำนวณ:**")
    st.write("1. ระบุจำนวนเดือนย้อนหลัง และกำหนดค่าน้ำหนัก (Weights) ให้แต่ละเดือน (ผลรวมน้ำหนักต้องเท่ากับ 1.0)")
    st.write("2. นำยอดเบิกของแต่ละเดือน คูณกับน้ำหนักที่กำหนดไว้ (มักให้น้ำหนักเดือนล่าสุดมากที่สุด)")
    st.write("3. รวมผลคูณของทุกเดือนเข้าด้วยกันเพื่อเป็นผลพยากรณ์")
    st.write("**เหมาะสำหรับ:** เมื่อเราเชื่อว่าเดือนที่ใกล้ที่สุดมีผลต่ออนาคตมากกว่าเดือนที่ไกลออกไป")

with st.sidebar.expander("4. Linear Regression (LR)"):
    st.write("**หลักการ:** สร้างเส้นตรงเพื่อหาแนวโน้ม (Trend) ของข้อมูล")
    st.latex(r"y = mx + c")
    st.write("**ขั้นตอนการคำนวณ:**")
    st.write("1. นำยอดเบิกในอดีตมาวางบนกราฟเพื่อดูแนวโน้ม")
    st.write("2. คำนวณหาเส้นตรงที่ลากผ่านจุดเหล่านั้นแล้วเกิดความคลาดเคลื่อนน้อยที่สุด")
    st.write("3. ใช้สมการเส้นตรงเพื่อลากเส้นต่อไปในอนาคต (เดือนถัดไป) เพื่อหาค่าพยากรณ์")
    st.write("**เหมาะสำหรับ:** ข้อมูลที่มีแนวโน้มเพิ่มขึ้นหรือลดลงอย่างชัดเจน")

with st.sidebar.expander("5. ARIMA"):
    st.write("**หลักการ:** โมเดลสถิติขั้นสูงที่ผสมผสาน 3 ส่วน:")
    st.write("- **AR (p):** การนำค่าในอดีตมาพยากรณ์ตัวมันเอง")
    st.write("- **I (d):** การหาผลต่างเพื่อให้ข้อมูลนิ่ง (Stationary)")
    st.write("- **MA (q):** การนำค่าความคลาดเคลื่อนในอดีตมาปรับปรุง")
    st.write("**สัมประสิทธิ์ (Coefficients):**")
    st.write("- **หาไปทำไม:** เพื่อหาน้ำหนักที่ 'เหมาะสมที่สุด' ของข้อมูลในอดีตและความผิดพลาดสะสม ที่ทำให้โมเดล 'เรียนรู้' พฤติกรรมการเบิกยาได้แม่นยำที่สุด")
    st.write("- **เอาไปใช้งานยังไง:** ระบบจะใช้ค่าเหล่านี้เป็นตัวคูณกับยอดการใช้ยาในอดีตและค่าความคลาดเคลื่อน เพื่อคำนวณออกมาเป็นตัวเลขพยากรณ์ในเดือนถัดไป")
    st.write("**ขั้นตอนการคำนวณ:**")
    st.write("1. **ส่วนต่าง (d):** ลบข้อมูลเดือนปัจจุบันกับเดือนก่อนเพื่อให้ข้อมูล 'นิ่ง' (กำจัดแนวโน้ม)")
    st.write("2. **AR (p):** คำนวณว่ายอดในอดีตส่งผลอย่างไรกับปัจจุบัน")
    st.write("3. **MA (q):** ดูว่าความผิดพลาดในอดีตควรถูกนำมาปรับจูนยอดในอนาคตเท่าใด")
    st.write("4. **รวมผล:** รวมทั้ง 3 ส่วนและบวกค่าส่วนต่างกลับเพื่อให้ได้ยอดพยากรณ์จริง")
    st.write("**เหมาะสำหรับ:** ข้อมูลที่มีความซับซ้อน มีรูปแบบความสัมพันธ์ของเวลาชัดเจน")

with st.sidebar.expander("📖 พจนานุกรมศัพท์พยากรณ์"):
    st.write("**- MAPE:** เปอร์เซ็นต์ความผิดพลาดเฉลี่ย ยิ่งน้อยยิ่งแม่นยำ (ควรน้อยกว่า 15-20%)")
    st.write("**- Stationary:** ข้อมูลที่นิ่ง คือข้อมูลที่มีค่าเฉลี่ยและส่วนเบี่ยงเบนคงที่ตลอดเวลา")
    st.write("**- p-value:** ค่าทางสถิติเพื่อตัดสินใจ ถ้า < 0.05 ใน ADF Test แสดงว่าข้อมูลนิ่ง")
    st.write("**- Residual:** ค่าจริง ลบด้วย ค่าพยากรณ์ (สิ่งที่โมเดลยังทำนายไม่ได้)")
    st.write("**- Autocorrelation:** ความสัมพันธ์ของข้อมูลปัจจุบันกับข้อมูลตัวเองในอดีต")
    st.write("**- CV (Coefficient of Variation):** ค่าสัมประสิทธิ์ความแปรผัน ใช้บอกความผันผวนของข้อมูลเทียบกับค่าเฉลี่ย ยิ่งสูงยิ่งพยากรณ์ยาก")
    st.write("**- Outlier:** ข้อมูลที่สูงหรือต่ำกว่าปกติอย่างมาก ซึ่งอาจส่งผลกระทบต่อความแม่นยำของโมเดล")
    st.write("**- Trend:** ทิศทางของข้อมูลในระยะยาว ว่ามีการเพิ่มขึ้นหรือลดลงอย่างต่อเนื่องหรือไม่")
    st.write("**- Seasonality:** รูปแบบของข้อมูลที่เกิดขึ้นซ้ำๆ ในช่วงเวลาที่แน่นอน เช่น ทุกๆ 6 หรือ 12 เดือน")
    st.write("**- Lag (การหน่วง):** การนำค่าในอดีตมาใช้ เช่น Lag 1 คือการใช้ข้อมูลของ 1 เดือนก่อนหน้า")

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

    # --- ส่วนที่ 2: Parameter Settings ---
    st.subheader("⚙️ การตั้งค่าพารามิเตอร์โมเดล")
    
    tabs = st.tabs(["📈 ARIMA Model", "📊 Moving Average", "📉 Linear Regression"])

    with tabs[0]:
        st.caption("ปรับแต่งค่า p, d, q สำหรับโมเดล ARIMA")
        col_p, col_d, col_q = st.columns(3)
        with col_p:
            p = st.number_input("p", min_value=0, max_value=12, value=6)
        with col_d:
            d = st.number_input("d", min_value=0, max_value=1, value=1)
        with col_q:
            q = st.number_input("q", min_value=0, max_value=12, value=1)

    with tabs[1]:
        st.caption("ตั้งค่าหน้าต่างเวลาและน้ำหนักสำหรับ MA/WMA")
        col_ma, col_wma = st.columns(2)
        with col_ma:
            ma_window = st.number_input("MA Window (Period)", min_value=1, max_value=12, value=6)
        with col_wma:
            wma_window = st.number_input("WMA Window", min_value=1, max_value=12, value=6)
        
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
        st.caption("ตั้งค่าช่วงข้อมูลสำหรับแนวโน้มเส้นตรง")
        lr_window = st.number_input("Linear Regression Window (0 = ทั้งหมด)", min_value=0, max_value=100, value=0)

    st.divider()

    # --- ส่วนที่ 3: Model Evaluation ---
    st.subheader("🚀 การประเมินประสิทธิภาพโมเดล")
    
    # ตรวจสอบว่ามีข้อมูลเดิมใน session หรือไม่ เพื่อไม่ให้ต้องรันใหม่เมื่อคุยกับ AI
    if "last_eval" not in st.session_state:
        st.session_state.last_eval = None

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
                    error_val = calculate_mape(actuals, all_preds[name])
                    mape_results.append({"Model": name, "MAPE (%)": f"{error_val:.2f}%"})
                    if error_val < min_mape:
                        min_mape = error_val
                        best_model_name = name
                
                # เตรียมข้อมูล MAPE results ในรูปแบบ List ของ Dictionary
                mape_list_to_save = mape_results # mape_results is already list of dicts

                # คำนวณ Residuals สำหรับทุกโมเดลเพื่อใช้ในระบบประเมินผล
                residual_dict = {}
                for name in model_names:
                    residual_dict[name] = [a - f for a, f in zip(actuals, all_preds[name])]

                # เก็บผลลัพธ์ไว้ใน session_state
                st.session_state.last_eval = {
                    "mape_results": mape_results,
                    "all_preds": all_preds,
                    "actuals": actuals,
                    "residual_dict": residual_dict,
                    "model_names": model_names,
                    "arima_label": arima_label,
                    "best_model_name": best_model_name,
                    "min_mape": min_mape,
                    "p": p, "d": d, "q": q,
                    "wma_weights": wma_weights
                }
                
                db_manager.save_model_config(selected_drug, {
                    "best_model": best_model_name,
                    "arima_order": [p, d, q],
                    "ma_window": ma_window,
                    "lr_window": lr_window,
                    "wma_weights": wma_weights,
                    "mape": min_mape,
                    "mape_results": mape_list_to_save
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

        # ส่วนการแสดงผลเดิม
        st.subheader("📊 ผลการประเมินโมเดล")
        col_table, col_chart = st.columns([1, 2])
        with col_table:
            st.write("ตาราง MAPE")
            st.table(pd.DataFrame(mape_results))
        
        with col_chart:
            st.write("📈 กราฟเปรียบเทียบ Actual vs Predictions")
            for name in model_names:
                # ปรับแต่ง Matplotlib ให้ดู Modern ขึ้น
                fig_eval, ax_eval = plt.subplots(figsize=(8, 3))
                months_label = [f"M+{j+1}" for j in range(len(actuals))]
                ax_eval.plot(months_label, actuals, label='Actual', color='#2c3e50', linewidth=2, marker='o')
                ax_eval.plot(months_label, all_preds[name], label=f'Predict', color='#3498db', linestyle='--', marker='x')
                ax_eval.set_title(f"Evaluation: {name}")
                ax_eval.legend()
                st.pyplot(fig_eval)

        st.success(f"✅ บันทึกโมเดลที่ดีที่สุดเรียบร้อย: {best_model_name} (MAPE: {min_mape:.2f}%)")

        # Insights และ Residual Analysis
        st.divider()
        st.subheader("🧠 ข้อมูลเชิงลึกของโมเดล")
        ins_col1, ins_col2 = st.columns(2)
        with ins_col1:
            st.write(f"**{arima_label} Coefficients**")
            arima_ins = ForecastModels.get_model_insights(pd.Series(current_demand), "ARIMA", order=(p, d, q))
            st.json(arima_ins)
        with ins_col2:
            st.write(f"**Weighted Moving Average Weights**")
            wma_df = pd.DataFrame({"Month Lag": [f"M-{len(wma_weights)-i}" for i in range(len(wma_weights))], "Weight": wma_weights})
            st.bar_chart(wma_df.set_index("Month Lag"))

        st.divider()
        st.subheader("🧐 การวิเคราะห์ค่าความคลาดเคลื่อน (Residual Analysis)")
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
    
    if config:
        st.info(f"💡 โมเดลที่แนะนำ (MAPE ต่ำที่สุด): **{config['best_model']}** ({config['mape']:.2f}%)")
        
        # ให้ User เลือกโมเดลที่ต้องการเอง
        saved_model_names = [res['Model'] for res in config.get('mape_results', [])]
        # Fallback กรณีข้อมูลเก่าไม่มีชื่อโมเดล
        if not saved_model_names:
            saved_model_names = ["Naive", "Moving Average", "Weighted Moving Average", "Linear Regression", "ARIMA"]
        
        # ค้นหา index ของโมเดลที่แนะนำเพื่อให้เป็นค่าเริ่มต้น
        try:
            default_idx = saved_model_names.index(config['best_model'])
        except ValueError:
            default_idx = 0
            
        selected_model_choice = st.selectbox("🎯 เลือกโมเดลที่ต้องการใช้งาน", saved_model_names, index=default_idx)
        
        # ปรับ required_len ตามโมเดลที่เลือก
        if "ARIMA" in selected_model_choice:
            required_len = config['arima_order'][0]
        elif "WMA" in selected_model_choice or "Weighted" in selected_model_choice:
            required_len = len(config['wma_weights'])
        elif "MA" in selected_model_choice or "Moving Average" in selected_model_choice:
            required_len = config.get('ma_window', 6)
        elif "LR" in selected_model_choice or "Linear Regression" in selected_model_choice:
            required_len = config.get('lr_window', 0)
            if required_len == 0: required_len = 12 # Default สำหรับ Full LR
        else: # Naive
            required_len = 1
    else:
        st.warning("ยังไม่มีการตั้งค่าโมเดลที่ดีที่สุดสำหรับยานี้ ระบบจะใช้ค่า Default")
        # ปรับให้ผู้ใช้สามารถเลือกโมเดลได้แม้ไม่มี config และตั้งค่าเริ่มต้นให้เหมาะสม
        selected_model_choice = st.selectbox("🎯 เลือกโมเดลที่ต้องการใช้งาน", ["Naive", "Moving Average", "Weighted Moving Average", "Linear Regression", "ARIMA"])
        
        if selected_model_choice == "Naive":
            required_len = 1
        else:
            required_len = 6

    st.subheader(f"📝 กรอกข้อมูลย้อนหลัง {required_len} เดือน สำหรับโมเดล {selected_model_choice}")
    
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
        # ปรับค่าเริ่มต้น (Fallback) ของ ARIMA ให้สอดคล้องกับค่า default required_len (6 เดือน)
        order = tuple(config['arima_order']) if config else (6,1,1)
        ma_win = config.get('ma_window', 6) if config else 6
        lr_win = config.get('lr_window', 0) if config else 0
        weights = config['wma_weights'] if config else [0.05, 0.1, 0.15, 0.2, 0.25, 0.25]
        
        # พยากรณ์เฉพาะโมเดลที่เลือก (หรือจะแสดงทั้งหมดแล้ว Highlight ก็ได้ แต่เพื่อความคลีนจะเน้นตัวที่เลือก)
        prediction = 0
        model_display_name = ""
        
        if "ARIMA" in selected_model_choice:
            prediction = ForecastModels.arima(history, order=order)
            model_display_name = f"ARIMA {order}"
        elif "Weighted" in selected_model_choice:
            prediction = ForecastModels.weighted_moving_average(history, weights=weights)
            model_display_name = "WMA"
        elif "Moving Average" in selected_model_choice:
            prediction = ForecastModels.moving_average(history, window=ma_win)
            model_display_name = f"MA ({ma_win}m)"
        elif "Linear Regression" in selected_model_choice:
            prediction = ForecastModels.linear_regression(history, window=lr_win if lr_win > 0 else None)
            model_display_name = "Linear Regression"
        else:
            prediction = ForecastModels.naive(history)
            model_display_name = "Naive"
            
        st.metric(label=f"🚀 ผลพยากรณ์เดือนถัดไป (t) ด้วยโมเดล {model_display_name}", value=f"{prediction:,.0f} หน่วย")
        
        with st.expander("ดูผลเปรียบเทียบจากโมเดลอื่น"):
            results_all = {
                "Naive": ForecastModels.naive(history),
                "Moving Average": ForecastModels.moving_average(history, window=ma_win),
                "WMA": ForecastModels.weighted_moving_average(history, weights=weights),
                "Linear Regression": ForecastModels.linear_regression(history, window=lr_win if lr_win > 0 else None),
                "ARIMA": ForecastModels.arima(history, order=order)
            }
            res_cols = st.columns(len(results_all))
            for i, (name, val) in enumerate(results_all.items()):
                res_cols[i].metric(name, f"{val:,.0f}")
