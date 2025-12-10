# edu_rec_sys/services/recommendation_service.py

import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ast
from collections import defaultdict
import requests
from django.conf import settings

# ⚠️ Pandas Warning Suppression (User Request)
pd.options.mode.chained_assignment = None


# 1단계에서 만든 모델 파일에서 클래스들을 가져옵니다.
from ..ml_models.transformer import (
    SharedEmbeddings, TermRecTransformer, Collator,
    build_samples_full_history, safe_int0, safe_float0
)

def filter_last_per_student(samples):
    """학생별로 가장 마지막 학기 샘플만 필터링하는 함수"""
    last = {}
    for s in samples:
        sid = s['student_id']
        if sid not in last or s['target_term'] > last[sid]['target_term']:
            last[sid] = s
    return list(last.values())

# --- 💡 중요: Singleton 패턴 ---
# RecommendationService 객체를 단 하나만 생성하여 메모리에 유지합니다.
class RecommendationService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RecommendationService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # 이미 초기화되었으면 다시 실행하지 않도록 방지
        if hasattr(self, 'initialized'):
            return
        
        print("🚀 RecommendationService 초기화를 시작합니다...")

        # --- 1. 경로 설정 ---
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.DATA_DIR = os.path.join(settings.BASE_DIR, 'edu_rec_sys', 'data')
        # 모델 및 임베딩 파일 경로
        self.MODEL_DIR = os.path.join(settings.BASE_DIR, 'sub')
        self.EMBEDDING_DIR = os.path.join(settings.BASE_DIR, 'sub')
        self.RAW_DATA_DIR = os.path.join(settings.BASE_DIR, 'edu_rec_sys', 'data')

        # --- 1.5 Checking and Downloading Files ---
        self._check_and_download_files()

        # --- 2. 전처리된 데이터 로드 ---
        self._load_preprocessed_data()
        
        # --- 3. 모델에 필요한 파라미터 계산 ---
        self._calculate_model_params()
        
        # --- 4. 모델 구조 정의 및 가중치 로드 ---
        self._load_model()
        
        # --- 5. 추천에 필요한 맵(map) 및 데이터 준비 ---
        self._prepare_prediction_assets()

        self.initialized = True
        self.initialized = True
        print("✅ RecommendationService 초기화 완료.")
        print("\n" + "="*50)
        print("✨✨✨ 모든 시스템 가동 준비 완료! (All Systems Operational) ✨✨✨")
        print("이제 웹사이트에 접속하여 학번을 입력하실 수 있습니다.")
        print("="*50 + "\n")

    def _check_and_download_files(self):
        """필요한 대용량 파일이 존재하는지 확인하고, 없으면 Dropbox에서 다운로드합니다."""
        files_to_check = [
            {
                "path": os.path.join(self.MODEL_DIR, "TermRecTransformer.pt"),
                "url": "https://www.dropbox.com/scl/fi/jv9ir3ekt0z0m9917u7vr/TermRecTransformer.pt?rlkey=4vatwhzykefbhv68454iuxo4m&dl=1",
                "name": "TermRecTransformer.pt"
            },
            {
                "path": os.path.join(self.EMBEDDING_DIR, "keyword_initial_embeddings.npy"),
                "url": "https://www.dropbox.com/scl/fi/ujlox6xk5v6bsysluqxjy/keyword_initial_embeddings.npy?rlkey=rwfpec4j9we3tt0f3qdg1yya7&dl=1",
                "name": "keyword_initial_embeddings.npy"
            },
            {
                "path": os.path.join(self.DATA_DIR, "df_student_grades_all.pkl"),
                "url": "https://www.dropbox.com/scl/fi/co3msjmaqwygi3x68ktx7/df_student_grades_all.pkl?rlkey=fnje3zi0a5xrerpf245bo1j6c&dl=1",
                "name": "df_student_grades_all.pkl"
            }
        ]

        print("📂 파일 무결성 및 다운로드 확인 시작...")
        for file_info in files_to_check:
            if not os.path.exists(file_info["path"]):
                print(f"  ⚠️ 파일이 없습니다: {file_info['name']}")
                print(f"  📥 다운로드 시작: {file_info['url']}")
                self._download_file(file_info["url"], file_info["path"])
                print(f"  ✅ 다운로드 완료: {file_info['name']}")
            else:
                print(f"  ✅ 파일 존재 확인: {file_info['name']}")

    def _download_file(self, url, dest_path):
        """URL에서 파일을 다운로드하여 저장합니다."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def _load_preprocessed_data(self):
        print("  - 1/5: 전처리된 데이터 로딩 중...")
        with open(os.path.join(self.DATA_DIR, 'encoders.pkl'), 'rb') as f:
            self.encoders = pickle.load(f)
        self.df_list_new_all = pd.read_pickle(os.path.join(self.DATA_DIR, 'df_list_new_all.pkl'))
        self.df_student_data = pd.read_pickle(os.path.join(self.DATA_DIR, 'df_student_data.pkl'))
        self.df_student_grades_all = pd.read_pickle(os.path.join(self.DATA_DIR, 'df_student_grades_all.pkl'))
        
        # 사용자 필터링에 필요한 2024년 1학기 데이터 로드
        self.list_sub_24 = pd.read_excel(os.path.join(self.RAW_DATA_DIR, "list_new.xlsx"))
        self.grade_sub_24 = pd.read_excel(os.path.join(self.RAW_DATA_DIR, "grade_new_all.xlsx"))
        self.how_sub_24 = pd.read_excel(os.path.join(self.RAW_DATA_DIR, "how_new_all.xlsx"))
        
        year, semester = 2024, 10
        self.list_sub_24 = self.list_sub_24[(self.list_sub_24['SYY'] == year) & (self.list_sub_24['SMT_DIV_CD'] == semester)]
        self.grade_sub_24 = self.grade_sub_24[(self.grade_sub_24['SYY'] == year) & (self.grade_sub_24['SMT_DIV_CD'] == semester)]
        self.how_sub_24 = self.how_sub_24[(self.how_sub_24['SYY'] == year) & (self.how_sub_24['SMT_DIV_CD'] == semester)]

        def parse_schedule(x):
            if isinstance(x, (list, tuple)): return x
            try: return ast.literal_eval(x) if pd.notna(x) else []
            except (ValueError, SyntaxError): return []
        self.list_sub_24['schedule_pairs'] = self.list_sub_24['schedule_pairs'].apply(parse_schedule)

    def _calculate_model_params(self):
        print("  - 2/5: 모델 파라미터 계산 중...")
        # 하이퍼파라미터
        self.D_MODEL, self.D_ID, self.D_META, self.D_TERM = 128, 32, 32, 16
        self.NHEAD, self.NLAYERS, self.D_FF, self.DROPOUT = 4, 2, 256, 0.3
        self.MAX_LEN_CAP = 150

        def get_max_encoded_val(df, col):
            v = df[col]
            return int(v[v.notna()].max()) if not v.empty else 0

        # Vocab 크기 계산
        self.num_courses = max(get_max_encoded_val(self.df_student_grades_all, 'SUBJTNB_encoded'),
                               get_max_encoded_val(self.df_list_new_all, 'SUBJTNB_encoded')) + 1
        self.num_students = self.df_student_data['ID'].max() + 1
        self.num_terms = int(self.df_student_grades_all['course_completed_year_term'].max()) + 1
        self.num_college = len(self.encoders['le_college'].classes_)
        self.num_major = len(self.encoders['le_major'].classes_)
        self.num_major_detail = len(self.encoders['le_md'].classes_)
        # ... (원본 코드의 모든 num_ 파라미터 계산) ...
        self.num_gen_type = get_max_encoded_val(self.df_list_new_all, 'general_type_id') + 1
        self.num_gen_subcat = get_max_encoded_val(self.df_list_new_all, 'general_subcategory_id') + 1
        self.num_gen_term = get_max_encoded_val(self.df_list_new_all, 'general_term_id') + 1
        self.num_subject_div = get_max_encoded_val(self.df_list_new_all, 'subject_div_id') + 1
        self.num_subj_cat = get_max_encoded_val(self.df_list_new_all, 'subject_category_id') + 1
        self.num_student_state = get_max_encoded_val(self.df_student_data, 'student_state_id') + 1
        self.num_su_yn = get_max_encoded_val(self.df_student_grades_all, 'su_id') + 1
        self.num_resit_yn = get_max_encoded_val(self.df_student_grades_all, 'retake_id') + 1
        self.num_transfer_type = get_max_encoded_val(self.df_student_data, 'transfer_type_id') + 1
        self.num_second_major = get_max_encoded_val(self.df_student_data, '2전공_id') + 1
        self.num_third_major = get_max_encoded_val(self.df_student_data, '3전공_id') + 1
        self.num_minor_major = get_max_encoded_val(self.df_student_data, '부전공_id') + 1
        self.num_second_minor_major = get_max_encoded_val(self.df_student_data, '2부전공_id') + 1
        self.num_micro_major = get_max_encoded_val(self.df_student_data, '마이크로전공_id') + 1
        self.num_entrance_major_dept = get_max_encoded_val(self.df_student_data, '입시학과_id') + 1
        self.num_grad_major_dept = get_max_encoded_val(self.df_student_data, '졸업학과_id') + 1

    def _load_model(self):
        print("  - 3/5: 모델 구조 정의 및 가중치 로딩 중...")
        kw_init = np.load(os.path.join(self.EMBEDDING_DIR, "keyword_initial_embeddings.npy"))
        theme_init = np.load(os.path.join(self.EMBEDDING_DIR, "theme_initial_embeddings.npy"))
        self.dim_kw_precomputed = kw_init.shape[1]
        self.dim_theme_precomputed = theme_init.shape[1]
        kw_tensor = torch.from_numpy(kw_init).float()
        theme_tensor = torch.from_numpy(theme_init).float()
        
        num_hist_cont_feats = 2

        shared_emb = SharedEmbeddings(
            d_id=self.D_ID, d_term=self.D_TERM, d_meta=self.D_META,
            num_students=self.num_students, num_courses=self.num_courses, num_terms=self.num_terms,
            num_college=self.num_college, num_major=self.num_major, num_major_detail=self.num_major_detail,
            num_gen_type=self.num_gen_type, num_gen_subcat=self.num_gen_subcat, num_gen_term=self.num_gen_term,
            num_subject_div=self.num_subject_div, num_subj_cat=self.num_subj_cat, num_su_yn=self.num_su_yn,
            num_resit_yn=self.num_resit_yn, num_student_state=self.num_student_state, num_second_major=self.num_second_major,
            num_third_major=self.num_third_major, num_minor_major=self.num_minor_major,
            num_second_minor_major=self.num_second_minor_major, num_micro_major=self.num_micro_major,
            num_entrance_major_dept=self.num_entrance_major_dept, num_grad_major_dept=self.num_grad_major_dept,
            num_transfer_type=self.num_transfer_type, dim_kw_precomputed=self.dim_kw_precomputed,
            dim_theme_precomputed=self.dim_theme_precomputed,
            initial_keyword_embeddings_tensor=kw_tensor,
            initial_theme_embeddings_tensor=theme_tensor,
            num_history_cont_feats=2 # 'course_completed_year_term', 'student_grade_score'
        )

        self.model = TermRecTransformer(
            shared_emb=shared_emb, 
            num_courses=self.num_courses,
            dim_kw_precomputed=self.dim_kw_precomputed,
            dim_theme_precomputed=self.dim_theme_precomputed,
            num_history_cont_feats=num_hist_cont_feats,
            d_model=self.D_MODEL,   
            nhead=self.NHEAD,        
            d_ff=self.D_FF,           
            nlayers=self.NLAYERS,      
            dropout=self.DROPOUT,      
            d_id=self.D_ID,            
            d_meta=self.D_META,        
            d_term=self.D_TERM,        
            use_positional=True
        )
        
        best_model_path = os.path.join(self.MODEL_DIR, "TermRecTransformer.pt")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.DEVICE))
        self.model.to(self.DEVICE)
        self.model.eval()


    def _prepare_prediction_assets(self):
            print("  - 4/5: 추론용 에셋 준비 중...")

            # [기존 로직 통합]
            # 전체 학생에 대한 이력 샘플 생성
            self.all_samples = build_samples_full_history(self.df_student_grades_all)
            self.all_samples_last = filter_last_per_student(self.all_samples)
            
            # [수정] 학생 ID로 샘플을 바로 찾기 위해 리스트 대신 딕셔너리로 캐시
            self.user_sample_info = {s['student_id']: (i, s) for i, s in enumerate(self.all_samples_last)}

            # 각종 변환 맵(map) 생성 및 저장
            self.encoded2subjtnb = {v: k for k, v in self.encoders['subjt_map'].items() if pd.notna(k) and pd.notna(v)}
            self.encoded2category = (
                self.df_list_new_all[['SUBJTNB_encoded', 'subject_category']]
                .drop_duplicates('SUBJTNB_encoded')
                .set_index('SUBJTNB_encoded')['subject_category']
                .to_dict()
            )
            self.encoded2div = (
                self.df_list_new_all[['SUBJTNB_encoded', 'subject_div']]
                .drop_duplicates('SUBJTNB_encoded')
                .set_index('SUBJTNB_encoded')['subject_div']
                .to_dict()
            )
            self.subjtnb2encoded = {v: k for k, v in self.encoded2subjtnb.items()}
            
            # 2024년 1학기 개설 과목 ID 집합 생성
            subjt_map = self.encoders['subjt_map']
            self.list_sub_24['SUBJTNB_encoded'] = self.list_sub_24['SUBJTNB'].map(subjt_map)
            self.offered_ids = set(self.list_sub_24['SUBJTNB_encoded'].dropna().astype(int).tolist())

            # Collator 인스턴스 생성
            max_len = min(max((len(s['hist_courses']) for s in self.all_samples), default=0), self.MAX_LEN_CAP)
            self.df_student_data_idx = self.df_student_data.set_index('ID')
            
            # course_meta_map 생성
            self.course_meta_map = {}
            for _, r in self.df_list_new_all.iterrows():
                cid = safe_int0(r['SUBJTNB_encoded'])
                self.course_meta_map[cid] = {
                    'college': safe_int0(r['college_id']), 'major': safe_int0(r['major_name_id']),
                    'major_detail': safe_int0(r['major_detail_id']), 'gen_type': safe_int0(r['general_type_id']),
                    'gen_subcat': safe_int0(r['general_subcategory_id']), 'gen_term': safe_int0(r['general_term_id']),
                    'subject_div': safe_int0(r['subject_div_id']), 'subject_category': safe_int0(r['subject_category_id']),
                    'difficulty': safe_float0(r['난이도_num']), 'evaluation': safe_float0(r['과목평점'])
                }
            
            # history_item_meta_map 생성
            self.history_item_meta_map = {}
            hist_df = self.df_student_grades_all[['ID', 'SUBJTNB_encoded', 'course_completed_year_term', 'su_id', 'retake_id', 'student_grade_score']]
            for _, r in hist_df.iterrows():
                key = (int(r['ID']), safe_int0(r['SUBJTNB_encoded']), int(r['course_completed_year_term']))
                self.history_item_meta_map[key] = {
                    'su_yn': safe_int0(r['su_id']), 'resit_yn': safe_int0(r['retake_id']),
                    'hist_cont_feats': np.array([safe_float0(r['course_completed_year_term']), safe_float0(r['student_grade_score'])], dtype=np.float32)
                }
            
            self.collate = Collator(
                C=self.num_courses, max_len=max_len, num_history_cont_feats=2,
                history_item_meta_map=self.history_item_meta_map, 
                course_meta_map=self.course_meta_map,
                df_student_data_idx=self.df_student_data_idx
            )
            print("  - 5/5: 서비스 준비 완료.")

    def get_all_predictions_for_student(self, student_id: int):
        """
        [분석용] 학생 ID를 받아 모든 개설 과목에 대한 원본 예측 점수를 DataFrame으로 반환합니다.
        (중복 분반 포함)
        """
        # 1. 학생의 마지막 학기 샘플 찾기 (self.user_sample_info는 dict)
        if student_id not in self.user_sample_info:
            print(f"\\n⚠️ ID {student_id}에 해당하는 학생 샘플이 없습니다.")
            return None
        
        _ , sample = self.user_sample_info[student_id]
        predicted_term = sample['target_term'] # ★★★ 수정된 부분 ★★★

        # 2. 모델 입력을 위한 데이터 준비 (self.collate 사용)
        (hist_ids, hist_terms, mask, stu_ids, _, _,
         course_meta, stu_meta) = self.collate([sample])

        # 3. 모델을 사용하여 예측 점수(logits) 계산 (self.model, self.DEVICE 사용)
        self.model.eval()
        with torch.no_grad():
            hist_ids = hist_ids.to(self.DEVICE)
            hist_terms = hist_terms.to(self.DEVICE)
            mask = mask.to(self.DEVICE)
            stu_ids = stu_ids.to(self.DEVICE)
            course_meta = {k: v.to(self.DEVICE) for k, v in course_meta.items()}
            stu_meta = {k: v.to(self.DEVICE) for k, v in stu_meta.items()}

            logits = self.model(hist_ids, hist_terms, mask, stu_ids,
                                course_meta=course_meta, stu_meta=stu_meta)
            
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        # 4. 예측 결과를 DataFrame에 매핑 (self.list_sub_24 사용)
        predictions_df = self.list_sub_24.copy()
        valid_courses = predictions_df.dropna(subset=['SUBJTNB_encoded']).copy()
        valid_courses['SUBJTNB_encoded'] = valid_courses['SUBJTNB_encoded'].astype(int)
        
        valid_courses['pred_score'] = valid_courses['SUBJTNB_encoded'].apply(
            lambda x: probs[x] if x < len(probs) else 0.0
        )

        sorted_df = valid_courses.sort_values(by='pred_score', ascending=False).reset_index(drop=True)
        return sorted_df, predicted_term 

    def predict_top_k(self, student_id: int):
        """
        학생 ID를 받아 모델을 통해 Top 80 과목을 추천합니다.
        get_all_predictions_for_student를 호출하여 결과를 처리합니다.
        """
        # 1. 헬퍼 함수를 호출하여 예측 결과와 학기 정보 받기
        prediction_result = self.get_all_predictions_for_student(student_id)

        if prediction_result is None:
            return None # 학생 정보가 없는 경우 None을 그대로 반환

        sorted_predictions_df, predicted_term = prediction_result        

        # 2. 중복 과목 제거 (과목 코드 기준)
        unique_predictions_df = sorted_predictions_df.drop_duplicates(subset=['SUBJTNB_encoded'])

        # 3. Top N 결정 및 추출
        N = 80
        top_n_df = unique_predictions_df.head(N)
        
        # 실제 추천된 과목 수로 N 업데이트
        N = len(top_n_df)

        if N == 0:
            print(f"⚠️ 서비스: 학생 ID {student_id}에게 추천할 과목이 없습니다.")
            return {
                "uid": student_id,
                "predicted_term": predicted_term,
                "N": 0, "topN_idx": [], "topN_subj": [], "topN_vals": []
            }

        # 4. 리스트 형태로 변환
        topN_idx_list = top_n_df['SUBJTNB_encoded'].tolist()
        topN_subj_list = top_n_df['SUBJTNB'].tolist()
        topN_vals_list = top_n_df['pred_score'].tolist()

        # 5. 최종 결과 딕셔너리 구성 및 반환 (기존 형식과 동일)
        result = {
            "uid": student_id,
            "predicted_term": predicted_term,
            "N": N,
            "topN_idx": topN_idx_list,
            "topN_subj": topN_subj_list,
            "topN_vals": topN_vals_list,
        }
        
        return result

    # [신규 추가] 학생 정보 및 전체 수강 이력을 가져오는 메서드
    def get_student_history(self, student_id: int):
        """학생의 기본 정보와 전체 학기 수강 이력을 딕셔너리 형태로 반환합니다."""
        student_info_df = self.df_student_data[self.df_student_data['ID'] == student_id]
        if student_info_df.empty:
            return None

        # 1. 학생 기본 정보 추출
        info = student_info_df.iloc[0].to_dict()

        # 2. 전체 수강 이력 추출 (숫자 학기 그대로 사용)
        student_courses_df = self.df_student_grades_all[self.df_student_grades_all['ID'] == student_id]
        
        if student_courses_df.empty:
            courses_by_term = {}
        else:
            # 숫자 학기를 기준으로 정렬 및 그룹화
            sorted_courses_df = student_courses_df.sort_values(by='course_completed_year_term')
            courses_by_term = sorted_courses_df.groupby('course_completed_year_term')['SUBJTNB'].apply(list).to_dict()
        
        # 템플릿에서 순서대로 표시하기 위해 (학기, 과목리스트) 튜플의 리스트로 변환
        sorted_history = sorted(courses_by_term.items())

        return {
            'info': info,
            'history': sorted_history
        }
    
    # [수정] 1단계: 모델 예측 결과를 "대표 분반" DataFrame으로 반환
    def predict_top_k_df(self, student_id: int):

        pred_result = self.predict_top_k(student_id)
        if not pred_result or not pred_result.get('topN_subj'):
            return pd.DataFrame()

        top_subjects = pd.DataFrame({
            'SUBJTNB': pred_result['topN_subj'],
            'pred_score': pred_result['topN_vals']
        })
        
        result_df = pd.merge(top_subjects, self.list_sub_24, on='SUBJTNB', how='inner')

        try:
            result_df['CORSE_DVCLS_NO_NUM'] = pd.to_numeric(result_df['CORSE_DVCLS_NO'])
            result_df = result_df.sort_values(['SUBJTNB', 'CORSE_DVCLS_NO_NUM'])
        except Exception:
            pass
            
        unique_result_df = result_df.drop_duplicates(subset='SUBJTNB', keep='first')
        return unique_result_df.sort_values('pred_score', ascending=False).reset_index(drop=True)

    def _normalize_choice_set(self, val, valid_universe=None, to_str=True):

        if val is None:
            return set()
        if isinstance(val, (list, tuple, set)):
            items = val
        else:
            items = [val]

        out = set()
        for x in items:
            try:
                s = str(x).strip() if to_str else x
            except Exception:
                continue
            if valid_universe is not None and s not in valid_universe:
                continue
            out.add(s)
        return out

    def _normalize_subject_category(self, val):

        valid = {'전공', '교양', '기타'}
        if val is None:
            return set()
        if isinstance(val, str):
            s = val.strip()
            return {s} if s in valid else set()
        if isinstance(val, (list, tuple, set)):
            return {str(x).strip() for x in val if str(x).strip() in valid}
        return set()

    def _safe_parse_pairs(self, pairs):

        if isinstance(pairs, (list, tuple)):
            out = []
            for p in pairs:
                try:
                    d, t = p
                    t = int(t)
                    out.append((str(d), t))
                except Exception:
                    return []
            return out
        if isinstance(pairs, str):
            try:
                obj = ast.literal_eval(pairs)
                return self._safe_parse_pairs(obj)
            except Exception:
                return []
        return []

    def filter_full_catalog(self, filter_criteria: dict):

        list_df = self.list_sub_24.copy()
        grade_df = self.grade_sub_24.copy()
        how_df = self.how_sub_24.copy()

        # [안전장치] 주요 문자열 컬럼 정리
        str_cols = [
            'subject_category', 'college_name', 'major_name', 'major_detail',
            'general_type', 'general_subcategory', 'general_term',
            'class_style', 'GRADE_EVL_MTHD_DIV_CD1'
        ]
        for col in str_cols:
            if col in list_df.columns:
                list_df[col] = list_df[col].astype(str).str.strip()

        # --- 1. subject_category 선택 처리 (문자열/리스트 모두 지원) ---
        raw_selected_category = filter_criteria.get('subject_category')
        categories_all = {'전공', '교양', '기타'}
        chosen = self._normalize_subject_category(raw_selected_category)
        student_category = chosen if chosen else categories_all

        # --- 2. 분기 처리: 단일 선택 시에만 하위 필터 활성화 ---
        # (다중 선택이면 상위 카테고리 필터만 적용)
        if student_category == {'전공'}:
            df_major_base = list_df[list_df['subject_category'] == '전공']

            college_universe = set(df_major_base['college_name'].dropna().astype(str).str.strip().unique())
            student_college = self._normalize_choice_set(filter_criteria.get('college_name'), valid_universe=college_universe)
            if not student_college:
                student_college = college_universe
            df_major_tmp = df_major_base[df_major_base['college_name'].isin(student_college)]

            major_universe = set(df_major_tmp['major_name'].dropna().astype(str).str.strip().unique())
            student_major = self._normalize_choice_set(filter_criteria.get('major_name'), valid_universe=major_universe)
            if not student_major:
                student_major = major_universe
            df_major_tmp2 = df_major_tmp[df_major_tmp['major_name'].isin(student_major)]

            detail_universe = set(df_major_tmp2['major_detail'].dropna().astype(str).str.strip().unique())
            student_major_detail = self._normalize_choice_set(filter_criteria.get('major_detail'), valid_universe=detail_universe)
            if not student_major_detail:
                student_major_detail = detail_universe

        elif student_category == {'교양'}:
            df_general_base = list_df[list_df['subject_category'] == '교양']

            gtype_universe = set(df_general_base['general_type'].dropna().astype(str).str.strip().unique())
            student_general_type = self._normalize_choice_set(filter_criteria.get('general_type_gyoyang'), valid_universe=gtype_universe)
            if not student_general_type:
                student_general_type = gtype_universe
            df_gen_tmp = df_general_base[df_general_base['general_type'].isin(student_general_type)]

            gsub_universe = set(df_gen_tmp['general_subcategory'].dropna().astype(str).str.strip().unique())
            student_general_subcat = self._normalize_choice_set(filter_criteria.get('general_subcategory_gyoyang'), valid_universe=gsub_universe)
            if not student_general_subcat:
                student_general_subcat = gsub_universe
            df_gen_tmp2 = df_gen_tmp[df_gen_tmp['general_subcategory'].isin(student_general_subcat)]

            gterm_universe = set(df_gen_tmp2['general_term'].dropna().astype(str).str.strip().unique())
            student_general_term = self._normalize_choice_set(filter_criteria.get('general_term_gyoyang'), valid_universe=gterm_universe)
            if not student_general_term:
                student_general_term = gterm_universe

        elif student_category == {'기타'}:
            df_etc_base = list_df[list_df['subject_category'] == '기타']
            etc_universe = set(df_etc_base['general_type'].dropna().astype(str).str.strip().unique())
            student_general_type = self._normalize_choice_set(filter_criteria.get('etc_type'), valid_universe=etc_universe)
            if not student_general_type:
                student_general_type = etc_universe

        # --- 3 & 4. 선호 교시 및 요일 처리 ---
        preferred_periods_list = filter_criteria.get('preferred_periods', [])
        preferred_days_list = filter_criteria.get('preferred_days', [])
        preferred_periods = set(map(int, preferred_periods_list)) if preferred_periods_list else set(range(16))
        preferred_days = set(preferred_days_list) if preferred_days_list else {'월', '화', '수', '목', '금', '토', '일'}

        # --- 5, 6, 7. 학점, 수업방식, 평가기준 처리 ---
        credit_list = filter_criteria.get('credit', [])
        if credit_list:
            # CDT가 float/str 혼재 대비
            try:
                preferred_credit = set(map(float, credit_list))
            except Exception:
                preferred_credit = set(credit_list)
        else:
            preferred_credit = set(list_df['CDT'].dropna().unique())

        class_styles_list = filter_criteria.get('class_styles', [])
        preferred_class_styles = set(class_styles_list) if class_styles_list else set(list_df['class_style'].dropna().unique())

        grade_eval_list = filter_criteria.get('grade_evaluation', [])
        grade_evaluation = set(grade_eval_list) if grade_eval_list else set(list_df['GRADE_EVL_MTHD_DIV_CD1'].dropna().unique())

        # --- 8 & 9. 성적 평가 방식 & 강의 방식 (복수 선택) 처리 ---
        available_grade_methods = ['중간', '기말', '퀴즈', '개인과제', '팀과제', '발표', '출석', '수업참여도', '추가1', '추가2', '추가3', '추가4']
        preferred_grade_methods = filter_criteria.get('grade_eval_methods') or available_grade_methods
        unselected_grade = [m for m in available_grade_methods if m not in preferred_grade_methods]

        # 키 컬럼 자료형/공백 방지
        key_cols = ['SYY', 'SMT_DIV_CD', 'SUBJTNB', 'CORSE_DVCLS_NO']
        for df in (grade_df, how_df, list_df):
            for kc in key_cols:
                if kc in df.columns:
                    df[kc] = df[kc].astype(str).str.strip()

        # grade_df 필터
        grade_use = grade_df.copy()
        for col in available_grade_methods:
            if col in grade_use.columns:
                grade_use[col] = pd.to_numeric(grade_use[col], errors='coerce')
        mask_grade = (
            (grade_use[preferred_grade_methods].fillna(0) >= 1).any(axis=1) &
            (grade_use[unselected_grade].fillna(0) == 0).all(axis=1)
        )
        grade_filtered = grade_use[mask_grade]

        # lecture/how_df 필터
        available_lecture_methods = ['강의', '실습', '발표', '토론', '팀프로젝트', '현장실습', '기타1', '기타2', '기타3']
        preferred_lecture_methods = filter_criteria.get('lecture_methods') or available_lecture_methods
        unselected_lecture = [m for m in available_lecture_methods if m not in preferred_lecture_methods]

        how_use = how_df.copy()
        for col in available_lecture_methods:
            if col in how_use.columns:
                how_use[col] = pd.to_numeric(how_use[col], errors='coerce')
        mask_lecture = (
            (how_use[preferred_lecture_methods].fillna(0) >= 1).any(axis=1) &
            (how_use[unselected_lecture].fillna(0) == 0).all(axis=1)
        )
        how_filtered = how_use[mask_lecture]

        # --- 최종 필터링(list_df 대상) ---
        df_filtered = list_df.copy()
        df_filtered = df_filtered[df_filtered['subject_category'].isin(student_category)]

        if student_category == {'전공'}:
            if filter_criteria.get('college_name'):
                df_filtered = df_filtered[df_filtered['college_name'].isin(student_college)]
            if filter_criteria.get('major_name'):
                df_filtered = df_filtered[df_filtered['major_name'].isin(student_major)]
            if filter_criteria.get('major_detail'):
                df_filtered = df_filtered[df_filtered['major_detail'].isin(student_major_detail)]

        elif student_category == {'교양'}:
            if filter_criteria.get('general_type_gyoyang'):
                df_filtered = df_filtered[df_filtered['general_type'].isin(student_general_type)]
            if filter_criteria.get('general_subcategory_gyoyang'):
                df_filtered = df_filtered[df_filtered['general_subcategory'].isin(student_general_subcat)]
            if filter_criteria.get('general_term_gyoyang'):
                df_filtered = df_filtered[df_filtered['general_term'].isin(student_general_term)]

        elif student_category == {'기타'}:
            if filter_criteria.get('etc_type'):
                df_filtered = df_filtered[df_filtered['general_type'].isin(student_general_type)]

        # 시간표 매칭
        def schedule_match(row_pairs):
            parsed = self._safe_parse_pairs(row_pairs)
            if not parsed:
                # 사용자가 요일/교시를 지정하지 않았다면 통과, 지정했다면 제외
                return not preferred_days_list and not preferred_periods_list
            # 모든 (요일,교시)가 선호 집합 내에 있어야 통과
            return all((d in preferred_days and t in preferred_periods) for d, t in parsed)

        if 'schedule_pairs' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['schedule_pairs'].apply(schedule_match)]

        # 기타 단일 선택 필터
        if 'CDT' in df_filtered.columns:
            # CDT가 str이면 float 비교가 안 될 수 있으므로 변환 시도
            try:
                df_filtered['__CDT_num__'] = pd.to_numeric(df_filtered['CDT'], errors='coerce')
                df_filtered = df_filtered[df_filtered['__CDT_num__'].isin(preferred_credit)]
                df_filtered.drop(columns=['__CDT_num__'], inplace=True)
            except Exception:
                df_filtered = df_filtered[df_filtered['CDT'].isin(preferred_credit)]

        if 'class_style' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['class_style'].isin(preferred_class_styles)]

        if 'GRADE_EVL_MTHD_DIV_CD1' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['GRADE_EVL_MTHD_DIV_CD1'].isin(grade_evaluation)]

        # --- [핵심] 최종 병합 로직 ---
        key_cols = ['SYY', 'SMT_DIV_CD', 'SUBJTNB', 'CORSE_DVCLS_NO']

        l24 = self.list_sub_24.copy()
        g24 = self.grade_sub_24.copy()
        h24 = self.how_sub_24.copy()
        for df0 in (l24, g24, h24):
            for kc in key_cols:
                if kc in df0.columns:
                    df0[kc] = df0[kc].astype(str).str.strip()

        for df1 in (df_filtered, grade_filtered, how_filtered):
            for kc in key_cols:
                if kc in df1.columns:
                    df1[kc] = df1[kc].astype(str).str.strip()

        merged_keys = pd.merge(
            df_filtered[key_cols].drop_duplicates(),
            grade_filtered[key_cols].drop_duplicates(),
            on=key_cols, how='inner'
        )
        merged_keys = pd.merge(
            merged_keys,
            how_filtered[key_cols].drop_duplicates(),
            on=key_cols, how='inner'
        )

        # 2) 공통 키로 원본(사본) 테이블 전체 정보 재결합
        final_df = pd.merge(merged_keys, l24, on=key_cols, how='inner')
        final_df = pd.merge(final_df, g24, on=key_cols, how='inner')
        final_df = pd.merge(final_df, h24, on=key_cols, how='inner')

        final_df = final_df.loc[:, ~final_df.columns.duplicated()].reset_index(drop=True)

        return final_df

    def get_filter_options(self):
        """HTML 템플릿에 전달할 모든 필터 옵션 목록을 생성합니다."""
        major_courses = self.list_sub_24[self.list_sub_24['subject_category'] == '전공'].copy()
        gyoyang_courses = self.list_sub_24[self.list_sub_24['subject_category'] == '교양'].copy()
        etc_courses = self.list_sub_24[self.list_sub_24['subject_category'] == '기타']

        etc_type_list = ['일반선택', '교직과정', '평생교육사과정']
        general_type_gyoyang_list = [
            gt for gt in gyoyang_courses['general_type'].dropna().unique()
            if gt not in etc_type_list
        ]

        options = {
            'subject_category': sorted(self.list_sub_24['subject_category'].dropna().unique()),
            'credit': sorted(self.list_sub_24['CDT'].dropna().unique()),
            'class_style': sorted(self.list_sub_24['class_style'].dropna().unique()),
            'grade_evaluation': sorted(self.list_sub_24['GRADE_EVL_MTHD_DIV_CD1'].dropna().unique()),
            'general_type_gyoyang': general_type_gyoyang_list,
            'general_term_gyoyang': sorted(gyoyang_courses['general_term'].dropna().unique()),
            'etc_type': etc_type_list,
            'days': ['월', '화', '수', '목', '금', '토', '일'],
            'periods': list(range(16)),
            'grade_eval_methods': ['중간', '기말', '퀴즈', '개인과제', '팀과제', '발표', '출석', '수업참여도', '추가1', '추가2', '추가3', '추가4'],
            'lecture_methods': ['강의', '실습', '발표', '토론', '팀프로젝트', '현장실습', '기타1', '기타2', '기타3']
        }

        # 전공 계층 구조
        for col in ['college_name', 'major_name', 'major_detail']:
            major_courses[col] = major_courses[col].fillna('N/A')
        major_hierarchy = {}
        unique_colleges = sorted([c for c in major_courses['college_name'].unique() if c != 'N/A'])
        options['college_name'] = unique_colleges
        for college in unique_colleges:
            major_hierarchy[college] = {}
            college_df = major_courses[major_courses['college_name'] == college]
            unique_majors = sorted([m for m in college_df['major_name'].unique() if m != 'N/A'])
            for major in unique_majors:
                major_df = college_df[college_df['major_name'] == major]
                details = sorted([d for d in major_df['major_detail'].unique() if d != 'N/A'])
                major_hierarchy[college][major] = details
        options['major_hierarchy'] = major_hierarchy

        # 교양 계층 구조
        for col in ['general_type', 'general_subcategory']:
            gyoyang_courses[col] = gyoyang_courses[col].fillna('N/A')
        gyoyang_hierarchy = {}
        for g_type in general_type_gyoyang_list:
            type_df = gyoyang_courses[gyoyang_courses['general_type'] == g_type]
            subcategories = sorted([sc for sc in type_df['general_subcategory'].unique() if sc != 'N/A'])
            gyoyang_hierarchy[g_type] = subcategories
        options['gyoyang_hierarchy'] = gyoyang_hierarchy

        return options

    def get_filtered_recommendations(self, student_id: int, filter_criteria: dict):
        """
        모델 추천 Top 60 과목과 사용자 필터링 결과의 교집합을 찾아 반환합니다.
        분반이 다르면 다른 과목으로 취급합니다.
        """
        # 1) 모델 추천 결과
        pred_result = self.predict_top_k(student_id)
        if not pred_result or not pred_result.get('topN_subj'):
            return pd.DataFrame()

        top_subjects_map = dict(zip(pred_result['topN_subj'], pred_result['topN_vals']))

        # 2) 사용자 필터 결과
        custom_filtered_df = self.filter_full_catalog(filter_criteria)

        if custom_filtered_df.empty:
            return pd.DataFrame()

        # 공백 제거/타입 통일
        custom_filtered_df['SUBJTNB'] = custom_filtered_df['SUBJTNB'].astype(str).str.strip()
        top_subjects_map_cleaned = {str(k).strip(): v for k, v in top_subjects_map.items()}

        # 3) 교집합
        intersection_df = custom_filtered_df[custom_filtered_df['SUBJTNB'].isin(top_subjects_map_cleaned.keys())].copy()
        if intersection_df.empty:
            return pd.DataFrame()

        # 4) 점수 부여 및 정렬
        intersection_df['pred_score'] = intersection_df['SUBJTNB'].map(top_subjects_map_cleaned)
        cols_front = [
            'SYY', 'SMT_DIV_CD', 'CAMPS_DIV_NM', 'seasonal_semester', 'subject_category',
            'college_name', 'major_name', 'major_detail', 'general_type', 'general_subcategory',
            'general_term', 'CDT', 'SUBJTNB', 'SUBJTNB_ENG', 'CORSE_DVCLS_NO', 'SUBJT_NM',
            'schedule_pairs', 'subject_div', 'L_ID', 'GRADE_EVL_MTHD_DIV_CD1', 'class_style',
            '난이도', 'keywords', '과목평점', 'theme1', 'theme2'
        ]
        cols_middle_grade = ['중간','기말','퀴즈','개인과제','팀과제','발표','출석','수업참여도','추가1','추가2','추가3','추가4']
        cols_middle_how = ['강의','실습','토론','팀프로젝트','현장실습','기타1','기타2','기타3']
        # 존재하는 컬럼만 골라 재정렬
        ordered = [c for c in cols_front + cols_middle_grade + cols_middle_how + ['pred_score'] if c in intersection_df.columns]
        intersection_df = intersection_df[ordered]

        return intersection_df.sort_values('pred_score', ascending=False).reset_index(drop=True)

# --- 서비스 인스턴스 생성 ---
recommendation_service = RecommendationService()
