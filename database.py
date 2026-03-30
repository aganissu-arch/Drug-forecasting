import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import json

class FirebaseManager:
    def __init__(self):
        
        self.db = self._init_firebase()

    def _init_firebase(self):
        """
        Health Check: ตรวจสอบการเชื่อมต่อ Firebase
        รองรับทั้งการรัน Local (ใช้ไฟล์ JSON) และ Streamlit Cloud (ใช้ Secrets)
        """
        if not firebase_admin._apps:
            try:
                # Sanity Check: ตรวจสอบว่ามี Secrets หรือไฟล์ Key หรือไม่
                if "firebase" in st.secrets:
                    # สำหรับ Streamlit Cloud
                    key_dict = dict(st.secrets["firebase"])
                    cred = credentials.Certificate(key_dict)
                else:
                    # สำหรับ Local Development (ระบุ Path ไปยังไฟล์ JSON ที่โหลดมา)
                    # แนะนำให้เก็บไฟล์ไว้ในที่ปลอดภัยและไม่นำขึ้น GitHub
                    cred = credentials.Certificate("serviceAccountKey.json")
                
                firebase_admin.initialize_app(cred)
                return firestore.client()
            except Exception as e:
                st.error(f"❌ Firebase Connection Health Check Failed: {e}")
                return None
        return firestore.client()

    def save_drug_data(self, drug_name, demand_list):
        """
        บันทึกข้อมูล Demand ลง Firestore
        """
        if self.db:
            try:
                doc_ref = self.db.collection('drug_inventory').document(drug_name)
                doc_ref.set({
                    'name': drug_name,
                    'demand': demand_list,
                    'last_updated': firestore.SERVER_TIMESTAMP
                })
                return True
            except Exception as e:
                st.error(f"Error saving to Firebase: {e}")
                return False
        return False

    def get_drug_data(self, drug_name):
        """
        Health Check: ดึงข้อมูล Demand จาก Firestore
        """
        if self.db:
            try:
                doc = self.db.collection('drug_inventory').document(drug_name).get()
                if doc.exists:
                    return doc.to_dict().get('demand', [])
            except Exception as e:
                st.error(f"Error fetching from Firebase: {e}")
        return []

    def get_all_drug_names(self):
        """
        ดึงรายชื่อยาทั้งหมดที่มีในระบบ
        """
        if self.db:
            try:
                docs = self.db.collection('drug_inventory').stream()
                return [doc.id for doc in docs]
            except Exception as e:
                st.error(f"Error fetching drug list: {e}")
        return []

    def delete_drug_data(self, drug_name):
        """
        ลบข้อมูลยาออกจาก Firestore
        """
        if self.db:
            try:
                self.db.collection('drug_inventory').document(drug_name).delete()
                return True
            except Exception as e:
                st.error(f"Error deleting from Firebase: {e}")
                return False
        return False

    def save_model_config(self, drug_name, config):
        """
        บันทึกค่าพารามิเตอร์ที่ดีที่สุดสำหรับยาแต่ละชนิด
        """
        if self.db:
            try:
                doc_ref = self.db.collection('model_configs').document(drug_name)
                doc_ref.set(config)
                return True
            except Exception as e:
                st.error(f"Error saving config: {e}")
                return False
        return False

    def get_model_config(self, drug_name):
        """
        ดึงค่าพารามิเตอร์ที่บันทึกไว้
        """
        if self.db:
            try:
                doc = self.db.collection('model_configs').document(drug_name).get()
                if doc.exists:
                    return doc.to_dict()
            except Exception as e:
                st.error(f"Error fetching config: {e}")
        return None
