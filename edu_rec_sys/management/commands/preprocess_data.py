# edu_rec_sys/management/commands/preprocess_data.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle  # 👈 파일 저장을 위해 추가

# Django 관리 명령어 기본 설정
from django.core.management.base import BaseCommand
from django.conf import settings # 👈 프로젝트의 settings.py에 접근하기 위해 추가

class Command(BaseCommand):
    help = 'Loads and preprocesses data for the recommendation model and saves the results.'

    # 👇 이 한 줄을 추가해주세요!
    requires_system_checks = []

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS('🚀 데이터 전처리 및 저장을 시작합니다...'))

        # ======================================================================
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 여기에 제공해주신 코드 전체를 붙여넣으세요 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # ======================================================================
        
        # 경로 설정 (사용자 환경에 맞게 수정 필요)
        base_path = os.path.join(settings.BASE_DIR, 'edu_rec_sys', 'data')
        
        # 파일 불러오기
        try:
            list_new = pd.read_excel(os.path.join(base_path, "list_new.xlsx"))
            student_data = pd.read_excel(os.path.join(base_path, "student_data.xlsx"))
            basic_new_words = pd.read_excel(os.path.join(base_path, "final_all_keywords.xlsx"))
            eval_df = pd.read_excel(os.path.join(base_path, "evaluation_final.xlsx"))
            theme = pd.read_excel(os.path.join(base_path, "basic_new(8055)_theme.xlsx"))
            how_all = pd.read_excel(os.path.join(base_path, "how_new_all.xlsx"))
            grade_all = pd.read_excel(os.path.join(base_path, "grade_new_all.xlsx"))
            part1 = pd.read_excel(os.path.join(base_path, "student_grades_all_part1.xlsx"))
            part2 = pd.read_excel(os.path.join(base_path, "student_grades_all_part2.xlsx"))
            student_grades_all = pd.concat([part1, part2], ignore_index=True)
            self.stdout.write("✅ 모든 데이터 파일을 성공적으로 불러왔습니다.")
        
        except FileNotFoundError as e:
            self.stderr.write(f"❌ 파일 로딩 오류: {e}. 'base_path' 변수의 경로를 확인해주세요.")
            return

        # 👇 여기에 빠진 코드를 추가해주세요!
        # --- 24년도 1학기 데이터 필터링 ---
        year = 2024
        semester = 10
        list_sub_24 = list_new[(list_new['SYY'] == year) & (list_new['SMT_DIV_CD'] == semester)].copy()
        how_sub_24 = how_all[(how_all['SYY'] == year) & (how_all['SMT_DIV_CD'] == semester)].copy()
        grade_sub_24 = grade_all[(grade_all['SYY'] == year) & (grade_all['SMT_DIV_CD'] == semester)].copy()
        # --- 여기까지 추가 ---

        # --- 2. 기본 전처리 ---
        common_ids = set(student_data['ID'].unique()) & set(student_grades_all['ID'].unique())

        student_grades_for_interaction = student_grades_all.copy()

        student_grades_for_interaction = student_grades_for_interaction.merge(
            student_data[['ID', 'student_college_name', 'student_major_name', 'student_major_detail']],
            on='ID', how='left'
        )
        unique_users = student_grades_all['ID'].unique().tolist()
        user2idx = {u: i for i, u in enumerate(unique_users)}

        # 공통으로 존재하는 ID만 필터링
        common_ids = set(student_data['ID'].unique()) & set(student_grades_all['ID'].unique())
        student_data = student_data[student_data['ID'].isin(common_ids)].reset_index(drop=True)

        A_student_data = student_data.copy()
        A_student_grades_all = student_grades_all.copy()

        print("✅ 기본 전처리가 완료되었습니다.")

        columns_to_keep = [
            'ID', 'course_completed_year_term', 'SYY', 'SMT_DIV_CD','SUBJT_NM',
            'college_name', 'major_name', 'major_detail',
            'general_type', 'general_subcategory', 'general_term',
            'SUBJTNB', 'CORSE_DVCLS_NO',
            'CDT', 'student_grade', '재수강여부', 'subject_div', '과목종별(수강)코드', 'SU여부'
        ]
        A_student_grades_all = A_student_grades_all[columns_to_keep]

        unique_terms = sorted(A_student_grades_all['course_completed_year_term'].unique(), key=lambda x: (
            int(x.split('-')[0]), int(x.split('-')[1][0])
        ))
        term_to_num = {term: i+1 for i, term in enumerate(unique_terms)}
        A_student_grades_all['course_completed_year_term'] = A_student_grades_all['course_completed_year_term'].map(term_to_num)

        grade_to_score = {
            'A+': 4.3, 'A0': 4.0, 'A-': 3.7, 'B+': 3.3, 'B0': 3.0, 'B-':2.7,
            'C+': 2.3, 'C0': 2.0, 'C-':1.7, 'D+': 1.3, 'D0': 1.0, 'D-':1.0,
            'F': 0.0, 'P': 4.3, 'NP': 0.0 , 'H' : 4.3, 'I': 0.0, 'nan': 0.0
        }
        A_student_grades_all['student_grade_score'] = A_student_grades_all['student_grade'].map(grade_to_score).fillna(0.0)

        duplicate_criteria = ['ID', 'course_completed_year_term', 'SUBJTNB']
        A_student_grades_all = A_student_grades_all.drop_duplicates(subset=duplicate_criteria, keep='first')

        print("✅ 과목 수강 이력 데이터 전처리가 완료되었습니다.")

        # 학생 정보 전처리
        A_student_data['입시학과'] = A_student_data['입시학과'].astype(str).apply(lambda x: '미래전공' if x.strip().endswith('(미래)') else x.strip())
        use_le_major_detail_cols = ['입시학과', '졸업학과', '2전공', '3전공', '4전공', '부전공', '2부전공', 'student_major_detail', '마이크로전공']
        sahak_map = {'사학(동양사분야)': '사학', '사학(서양사분야)': '사학', '사학(한국사분야)': '사학'}
        for col in use_le_major_detail_cols:
            if col in A_student_data.columns:
                A_student_data[col] = A_student_data[col].astype(str).replace(sahak_map)

        # 과목 정보 전처리
        list_new_all = list_new.copy()
        difficulty_map = {'쉬움': 1, '쉬움 (표본 부족)': 1, '보통': 2, '보통 (표본 부족)': 2, '어려움': 3, '어려움 (표본 부족)': 3}
        list_new_all['난이도_num'] = list_new_all['난이도'].map(difficulty_map).fillna(2).astype(int)

        # list_new에 없는 과목에 대한 더미 행 생성
        only_in_grades = set(A_student_grades_all['SUBJTNB']) - set(list_new_all['SUBJTNB'])
        CAT_EXTRA = ['college_name','major_name','major_detail','general_type','general_subcategory','general_term','subject_div','CDT']
        grades_meta_latest = A_student_grades_all.loc[A_student_grades_all['SUBJTNB'].isin(only_in_grades), ['SUBJTNB','SYY','SMT_DIV_CD'] + CAT_EXTRA].sort_values(['SYY','SMT_DIV_CD']).groupby('SUBJTNB', as_index=False).tail(1).set_index('SUBJTNB')

        dummy_rows = []
        for subj in only_in_grades:
            if subj in grades_meta_latest.index:
                meta = grades_meta_latest.loc[subj]
                row_data = {'SUBJTNB': subj, '난이도_num': 2, 'GRADE_EVL_MTHD_DIV_CD1': '절대평가', 'class_style': '대면강의', 'subject_category': '전공'}
                row_data.update({k: meta[k] for k in CAT_EXTRA})
                dummy_rows.append(row_data)
        dummy_df = pd.DataFrame(dummy_rows)

        course_cols_needed = ['SUBJTNB','college_name','major_name','major_detail','general_type','general_subcategory','general_term','subject_div','난이도_num','GRADE_EVL_MTHD_DIV_CD1','class_style','subject_category']
        list_new_all = pd.concat([list_new_all.reindex(columns=course_cols_needed), dummy_df.reindex(columns=course_cols_needed)], ignore_index=True)
        list_new_all['난이도_num'] = list_new_all['난이도_num'].fillna(2).astype(int)
        for col, default in [('GRADE_EVL_MTHD_DIV_CD1', '절대평가'), ('class_style', '대면강의'), ('subject_category', '전공')]:
            list_new_all[col] = list_new_all[col].fillna(default)
        list_new_all = list_new_all.drop_duplicates(subset=['SUBJTNB'], keep='first').reset_index(drop=True)

        # 키워드, 평가, 테마 정보 병합
        list_new_all = list_new_all.merge(basic_new_words[['SUBJTNB', 'keywords']], on='SUBJTNB', how='left')
        list_new_all = list_new_all.merge(eval_df, on='SUBJTNB', how='left')
        list_new_all['과목평점'] = list_new_all['과목평점'].fillna(2.56)
        list_new_all = list_new_all.merge(theme, on='SUBJTNB', how='left')

        # 키워드, 평가, 테마 정보 병합
        list_sub_24 = list_sub_24.merge(basic_new_words[['SUBJTNB', 'keywords']], on='SUBJTNB', how='left')
        list_sub_24 = list_sub_24.merge(eval_df, on='SUBJTNB', how='left')
        list_sub_24['과목평점'] = list_sub_24['과목평점'].fillna(2.56)
        list_sub_24 = list_sub_24.merge(theme, on='SUBJTNB', how='left')

        print("✅ 학생 및 과목 상세 정보 전처리가 완료되었습니다.")

        # 모델에 사용할 최종 데이터프레임 선택
        df_student_grades_all = A_student_grades_all[['ID','course_completed_year_term','SUBJTNB','재수강여부','SU여부','student_grade_score']].copy()
        df_student_data = A_student_data[['ID','student_college_name','student_major_name','student_major_detail','입학년월','student_state','신편입구분','입시학과','졸업학과','2전공','3전공','부전공','2부전공','마이크로전공']].copy()
        df_list_new_all = list_new_all[['SUBJTNB','college_name','major_name','major_detail','general_type','general_subcategory','general_term','subject_div','난이도_num','subject_category','keywords', '과목평점', 'theme1', 'theme2']].copy()

        # SUBJTNB 인코딩
        all_subjt_codes = pd.concat([df_student_grades_all['SUBJTNB'], df_list_new_all['SUBJTNB']]).unique()
        subjt_map = {code: i for i, code in enumerate(all_subjt_codes)}
        df_student_grades_all['SUBJTNB_encoded'] = df_student_grades_all['SUBJTNB'].map(subjt_map)
        df_list_new_all['SUBJTNB_encoded'] = df_list_new_all['SUBJTNB'].map(subjt_map)
        NUM_TOTAL_COURSES = len(all_subjt_codes)

        def label_encode_with_na(series: pd.Series, le=None):
            placeholder = 'Unknown'
            if le is None:
                le = LabelEncoder().fit(series.fillna(placeholder))

            encoded = pd.Series(pd.NA, index=series.index, dtype="Int64")
            mask = series.notna()
            # .loc을 사용하여 boolean indexing으로 값 할당
            if mask.any():
                encoded.loc[mask] = le.transform(series[mask])
            return encoded, le
        
        # 카테고리 피처 인코딩
        placeholder = 'Unknown'
        all_colleges = pd.concat([df_list_new_all['college_name'].fillna(placeholder), df_student_data['student_college_name'].fillna(placeholder)])
        le_college = LabelEncoder().fit(all_colleges)
        df_list_new_all['college_id'], _ = label_encode_with_na(df_list_new_all['college_name'], le_college)
        df_student_data['student_college_id'], _ = label_encode_with_na(df_student_data['student_college_name'], le_college)

        all_majors = pd.concat([df_list_new_all['major_name'].fillna(placeholder), df_student_data['student_major_name'].fillna(placeholder)])
        le_major = LabelEncoder().fit(all_majors)
        df_list_new_all['major_name_id'], _ = label_encode_with_na(df_list_new_all['major_name'], le_major)
        df_student_data['student_major_name_id'], _ = label_encode_with_na(df_student_data['student_major_name'], le_major)

        major_detail_cols = ['major_detail', 'student_major_detail', '입시학과', '졸업학과', '2전공', '3전공', '부전공', '2부전공', '마이크로전공']
        all_major_details = pd.concat([df_list_new_all['major_detail'].fillna(placeholder)] + [df_student_data[col].fillna(placeholder) for col in major_detail_cols if col in df_student_data.columns])
        le_md = LabelEncoder().fit(all_major_details)
        df_list_new_all['major_detail_id'], _ = label_encode_with_na(df_list_new_all['major_detail'], le_md)
        for col in major_detail_cols:
            if col in df_student_data.columns:
                df_student_data[col + '_id'], _ = label_encode_with_na(df_student_data[col], le_md)

        # 기타 카테고리 피처 인코딩
        simple_encode_cols_grades = ['재수강여부', 'SU여부']
        simple_encode_cols_student = ['student_state', '신편입구분']
        simple_encode_cols_list = ['subject_div', 'subject_category', 'general_type', 'general_subcategory', 'general_term']

        for col in simple_encode_cols_grades:
            df_student_grades_all[col + '_id'], _ = label_encode_with_na(df_student_grades_all[col])
        for col in simple_encode_cols_student:
            df_student_data[col + '_id'], _ = label_encode_with_na(df_student_data[col])
        for col in simple_encode_cols_list:
            df_list_new_all[col + '_id'], _ = label_encode_with_na(df_list_new_all[col])

        print("✅ 최종 데이터프레임 생성 및 모든 피처 인코딩이 완료되었습니다.")

        # --- general_term 인코딩 ---
        all_subjt_codes = pd.concat([df_student_grades_all['SUBJTNB'], df_list_new_all['SUBJTNB']]).unique()
        subjt_map = {code: i for i, code in enumerate(all_subjt_codes)}
        df_student_grades_all['SUBJTNB_encoded'] = df_student_grades_all['SUBJTNB'].map(subjt_map)
        df_list_new_all['SUBJTNB_encoded'] = df_list_new_all['SUBJTNB'].map(subjt_map)
        NUM_TOTAL_COURSES = len(all_subjt_codes)

        
        # list_sub_24에도 동일한 매핑 적용
        list_sub_24['SUBJTNB_encoded'] = list_sub_24['SUBJTNB'].map(subjt_map)
        # 1) 누락값을 처리할 플레이스홀더 정의
        placeholder = 'Unknown'

        # 2) 모델에 피팅할 전체 카테고리 시리즈 생성 (결측 → placeholder)
        all_colleges = pd.concat([
            df_list_new_all['college_name'].fillna(placeholder),
            df_student_data['student_college_name'].fillna(placeholder)
        ], ignore_index=True)

        # 3) LabelEncoder 학습
        le_college = LabelEncoder()
        le_college.fit(all_colleges)

        # 4) 변환된 ID 컬럼 생성 (원본 결측은 그대로 <NA>로 유지)
        #    — 먼저 전체를 <NA>로 초기화한 뒤, notna()인 부분만 변환
        df_list_new_all['college_id'] = pd.Series(pd.NA, index=df_list_new_all.index, dtype='Int64')
        mask1 = df_list_new_all['college_name'].notna()
        df_list_new_all.loc[mask1, 'college_id'] = le_college.transform(
            df_list_new_all.loc[mask1, 'college_name']
        )

        df_student_data['student_college_id'] = pd.Series(pd.NA, index=df_student_data.index, dtype='Int64')
        mask2 = df_student_data['student_college_name'].notna()
        df_student_data.loc[mask2, 'student_college_id'] = le_college.transform(
            df_student_data.loc[mask2, 'student_college_name']
        )

        # 1) 결측값 처리용 플레이스홀더
        placeholder = 'Unknown'

        # 2) LabelEncoder 학습용 전체 시리즈 준비
        all_majors = pd.concat([
            df_list_new_all['major_name'].fillna(placeholder),
            df_student_data['student_major_name'].fillna(placeholder)
        ], ignore_index=True)

        le_major = LabelEncoder()
        le_major.fit(all_majors)

        # 3) df_list_new_all에 major_name_id 생성 (원본 NaN은 <NA>로 남김)
        df_list_new_all['major_name_id'] = pd.Series(pd.NA, index=df_list_new_all.index, dtype='Int64')
        mask_list = df_list_new_all['major_name'].notna()
        df_list_new_all.loc[mask_list, 'major_name_id'] = le_major.transform(
            df_list_new_all.loc[mask_list, 'major_name']
        )

        # 4) df_student_data에 student_major_name_id 생성 (원본 NaN은 <NA>로 남김)
        df_student_data['student_major_name_id'] = pd.Series(pd.NA, index=df_student_data.index, dtype='Int64')
        mask_stud = df_student_data['student_major_name'].notna()
        df_student_data.loc[mask_stud, 'student_major_name_id'] = le_major.transform(
            df_student_data.loc[mask_stud, 'student_major_name']
        )

        # 1) 처리할 컬럼 리스트 정의
        major_detail_related_cols = [
            'major_detail', 'student_major_detail', '입시학과', '졸업학과',
            '2전공', '3전공', '부전공', '2부전공', '마이크로전공'
        ]

        # 2) 결측 처리용 플레이스홀더
        placeholder = 'Unknown'

        # 3) 학습용 전체 시리즈 생성 (fillna → placeholder)
        all_major_details = pd.concat(
            [
                df_list_new_all[col].fillna(placeholder)
                for col in major_detail_related_cols
                if col in df_list_new_all.columns
            ] + [
                df_student_data[col].fillna(placeholder)
                for col in major_detail_related_cols
                if col in df_student_data.columns
            ],
            ignore_index=True
        )

        # 4) LabelEncoder 학습
        le_md = LabelEncoder()
        le_md.fit(all_major_details)

        # 5) df_list_new_all에 major_detail_id 생성 (<NA> 유지)
        df_list_new_all['major_detail_id'] = pd.Series(pd.NA, index=df_list_new_all.index, dtype='Int64')
        mask = df_list_new_all['major_detail'].notna()
        df_list_new_all.loc[mask, 'major_detail_id'] = le_md.transform(
            df_list_new_all.loc[mask, 'major_detail']
        )

        # 6) df_student_data에 각 컬럼별 ID 생성 (<NA> 유지)
        for src_col, dst_col in [
            ('student_major_detail', 'student_major_detail_id'),
            ('입시학과',              'admission_dept_id'),
            ('졸업학과',              'graduation_dept_id'),
            ('2전공',               'major_2_id'),
            ('3전공',               'major_3_id'),
            ('부전공',               'minor_1_id'),
            ('2부전공',              'minor_2_id'),
            ('마이크로전공',          'micro_major_id'),
        ]:
            df_student_data[dst_col] = pd.Series(pd.NA, index=df_student_data.index, dtype='Int64')
            mask = df_student_data[src_col].notna()
            df_student_data.loc[mask, dst_col] = le_md.transform(
                df_student_data.loc[mask, src_col]
            )

        # --- 1. 재수강여부 (df_student_grades_all) ---
        df_student_grades_all['retake_id'], le_retake = label_encode_with_na(
            df_student_grades_all['재수강여부']
        )

        # --- 2. SU여부 (df_student_grades_all) ---
        df_student_grades_all['su_id'], le_su = label_encode_with_na(
            df_student_grades_all['SU여부']
        )

        # --- 3. student_state (df_student_data) ---
        df_student_data['student_state_id'], le_state = label_encode_with_na(
            df_student_data['student_state']
        )

        # --- 4. 신편입구분 (df_student_data) ---
        df_student_data['transfer_type_id'], le_transfer = label_encode_with_na(
            df_student_data['신편입구분']
        )

        # --- 6. subject_div (df_list_new_all) ---
        df_list_new_all['subject_div_id'], le_subdiv = label_encode_with_na(
            df_list_new_all['subject_div']
        )

        # --- 7. subject_category (df_list_new_all) ---
        df_list_new_all['subject_category_id'], le_subcat = label_encode_with_na(
            df_list_new_all['subject_category']
        )

        # --- general_type 인코딩 ---
        df_list_new_all['general_type_id'], le_gen_type = label_encode_with_na(
            df_list_new_all['general_type']
        )

        # --- general_subcategory 인코딩 ---
        df_list_new_all['general_subcategory_id'], le_gen_subcat = label_encode_with_na(
            df_list_new_all['general_subcategory']
        )

        # --- general_term 인코딩 ---
        df_list_new_all['general_term_id'], le_gen_term = label_encode_with_na(
            df_list_new_all['general_term']
        )

        self.stdout.write("✅ 최종 데이터프레임 생성 및 모든 피처 인코딩이 완료되었습니다.")        
        
        # 저장할 폴더 경로 설정 (edu_rec_sys/data/)
        output_dir = os.path.join(settings.BASE_DIR, 'edu_rec_sys', 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 데이터프레임 저장
        df_student_grades_all.to_pickle(os.path.join(output_dir, 'df_student_grades_all.pkl'))
        df_student_data.to_pickle(os.path.join(output_dir, 'df_student_data.pkl'))
        df_list_new_all.to_pickle(os.path.join(output_dir, 'df_list_new_all.pkl'))
        self.stdout.write(self.style.SUCCESS('💾 데이터프레임 저장 완료!'))

        # 2. LabelEncoder 및 맵핑 객체 저장
        encoders = {
            'subjt_map': subjt_map,
            'le_college': le_college,
            'le_major': le_major,
            'le_md': le_md,
            'le_retake': le_retake,
            'le_su': le_su,
            'le_state': le_state,
            'le_transfer': le_transfer,
            'le_subdiv': le_subdiv,
            'le_subcat': le_subcat,
            'le_gen_type': le_gen_type,
            'le_gen_subcat': le_gen_subcat,
            'le_gen_term': le_gen_term,
        }
        
        with open(os.path.join(output_dir, 'encoders.pkl'), 'wb') as f:
            pickle.dump(encoders, f)
        
        self.stdout.write(self.style.SUCCESS('💾 인코더(LabelEncoders) 저장 완료!'))
        self.stdout.write(self.style.SUCCESS('🎉 모든 작업이 성공적으로 완료되었습니다.'))
