# edu_rec_sys/ml_models/transformer.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# ======================================================================
# 헬퍼 함수 및 클래스 (데이터 처리용)
# ======================================================================

def safe_int0(x):
    """NaN이 아닌 경우 int로 변환하고, NaN이면 0을 반환합니다."""
    return int(x) if pd.notna(x) else 0

def safe_float0(x):
    """NaN이 아닌 경우 float로 변환하고, NaN이면 0.0을 반환합니다."""
    return float(x) if pd.notna(x) else 0.0

def build_samples_full_history(df: pd.DataFrame, id_col='ID', course_col='SUBJTNB_encoded', term_col='course_completed_year_term'):
    """학생별 수강 이력 전체를 사용하여 다음 학기 수강 과목을 예측하는 샘플을 생성합니다."""
    samples = []
    for sid, g in df.groupby(id_col):
        g = g.sort_values(term_col)
        terms = g[term_col].values
        courses = g[course_col].values
        unique_terms = np.unique(terms)
        
        # 첫 학기는 예측 대상이 아니므로 건너뜁니다.
        for t in unique_terms[1:]:
            hist_mask = terms < t
            target_mask = terms == t
            hist_courses = courses[hist_mask]
            target_courses = courses[target_mask]

            if len(target_courses) > 0 and len(hist_courses) > 0:
                samples.append({
                    'student_id': int(sid),
                    'hist_courses': hist_courses.tolist(),
                    'hist_terms': terms[hist_mask].tolist(),
                    'target_courses': target_courses.tolist(),
                    'target_term': int(t)
                })
    return samples

class CourseRecDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class Collator:
    def __init__(self,
                 C: int,
                 max_len: int,
                 num_history_cont_feats: int,
                 history_item_meta_map: dict,
                 course_meta_map: dict,
                 df_student_data_idx):
        self.C = C
        self.max_len = max_len
        self.num_history_cont = num_history_cont_feats
        self.hist_map = history_item_meta_map
        self.course_meta_map = course_meta_map
        self.df_student_data_idx = df_student_data_idx

    def __call__(self, batch):
        B = len(batch)

        # 1) 시퀀스 인덱스 + mask
        hist_ids   = np.zeros((B, self.max_len), dtype=np.int64)
        hist_terms = np.zeros((B, self.max_len), dtype=np.int64)
        mask       = np.zeros((B, self.max_len), dtype=bool)

        # 2) 학생 ID, target term, multi-hot labels
        stu_ids   = np.zeros(B, dtype=np.int64)
        tgt_terms = np.zeros(B, dtype=np.int64)
        labels    = np.zeros((B, self.C), dtype=np.float32)

        # 3) 과목 메타: categorical
        college      = np.zeros((B, self.max_len), dtype=np.int64)
        major        = np.zeros((B, self.max_len), dtype=np.int64)
        major_detail = np.zeros((B, self.max_len), dtype=np.int64)
        gen_type     = np.zeros((B, self.max_len), dtype=np.int64)
        gen_subcat   = np.zeros((B, self.max_len), dtype=np.int64)
        gen_term     = np.zeros((B, self.max_len), dtype=np.int64)
        subject_div  = np.zeros((B, self.max_len), dtype=np.int64)
        subj_cat     = np.zeros((B, self.max_len), dtype=np.int64)

        # 3‑1) 과목 메타: continuous
        difficulty   = np.zeros((B, self.max_len), dtype=np.float32)
        evaluation     = np.zeros((B, self.max_len), dtype=np.float32)

        # 4) 이력 특화 categorical
        su_yn_arr       = np.zeros((B, self.max_len), dtype=np.int64)
        resit_yn_arr    = np.zeros((B, self.max_len), dtype=np.int64)

        # 4‑1) 이력 특화 continuous
        hist_cont_feats = np.zeros((B, self.max_len, self.num_history_cont), dtype=np.float32)

        # 5) 학생 메타 categorical
        stu_college      = np.zeros(B, dtype=np.int64)
        stu_major        = np.zeros(B, dtype=np.int64)
        stu_major_detail = np.zeros(B, dtype=np.int64)
        stu_state        = np.zeros(B, dtype=np.int64)
        stu_second_major        = np.zeros(B, dtype=np.int64)
        stu_third_major         = np.zeros(B, dtype=np.int64)
        stu_minor_major         = np.zeros(B, dtype=np.int64)
        stu_second_minor_major  = np.zeros(B, dtype=np.int64)
        stu_micro_major         = np.zeros(B, dtype=np.int64)
        stu_entrance_major_dept = np.zeros(B, dtype=np.int64)
        stu_grad_major_dept     = np.zeros(B, dtype=np.int64)
        stu_transfer_type       = np.zeros(B, dtype=np.int64)

        # --- fill ---
        for i, s in enumerate(batch):
            sid = s['student_id']
            row = self.df_student_data_idx.loc[sid]

            # (A) student meta
            stu_college[i]       = row['student_college_id']
            stu_major[i]         = row['student_major_name_id']
            stu_major_detail[i]  = row['student_major_detail_id']
            stu_state[i]         = row['student_state_id']
            stu_second_major[i]        = row['major_2_id']
            stu_third_major[i]         = row['major_3_id']
            stu_minor_major[i]         = row['minor_1_id']
            stu_second_minor_major[i]  = row['minor_2_id']
            stu_micro_major[i]         = row['micro_major_id']
            stu_entrance_major_dept[i] = row['admission_dept_id']
            stu_grad_major_dept[i]     = row['graduation_dept_id']
            stu_transfer_type[i]       = row['transfer_type_id']

            stu_ids[i]   = sid
            tgt_terms[i] = s['target_term']

            # (B) history sequence
            h_ids   = s['hist_courses'][-self.max_len:]
            h_terms = s['hist_terms'][-self.max_len:]
            L = len(h_ids)
            hist_ids[i, :L]   = h_ids
            hist_terms[i, :L] = h_terms
            mask[i, :L]       = True

            for cid in s['target_courses']:
                labels[i, cid] = 1.0

            # (C) per-item meta
            for t_idx, cid in enumerate(h_ids):
                meta = self.course_meta_map[int(cid)]

                # categorical
                college[i, t_idx]      = meta['college']
                major[i, t_idx]        = meta['major']
                major_detail[i, t_idx] = meta['major_detail']
                gen_type[i, t_idx]     = meta['gen_type']
                gen_subcat[i, t_idx]   = meta['gen_subcat']
                gen_term[i, t_idx]     = meta['gen_term']
                subject_div[i, t_idx]  = meta['subject_div']
                subj_cat[i, t_idx]     = meta['subject_category']

                # continuous
                difficulty[i, t_idx]   = meta['difficulty']
                evaluation[i, t_idx]     = meta['evaluation']

                # history-specific
                hk = (sid, int(cid), int(h_terms[t_idx]))
                hmeta = self.hist_map.get(hk)
                if hmeta:
                    su_yn_arr[i, t_idx]      = hmeta['su_yn']
                    resit_yn_arr[i, t_idx]   = hmeta['resit_yn']
                    hist_cont_feats[i, t_idx] = hmeta['hist_cont_feats']

        for arr in [college, major, major_detail,
                    gen_type, gen_subcat, gen_term,
                    subject_div, subj_cat,
                    su_yn_arr, resit_yn_arr]:
            arr[arr < 0] = 0

        for arr in [stu_college, stu_major, stu_major_detail, stu_state,
                    stu_second_major, stu_third_major, stu_minor_major,
                    stu_second_minor_major, stu_micro_major,
                    stu_entrance_major_dept, stu_grad_major_dept,
                    stu_transfer_type]:
            arr[arr < 0] = 0
            
        hist_ids[hist_ids < 0]     = 0
        hist_terms[hist_terms < 0] = 0
        course_meta = {
            'college':          torch.from_numpy(college),
            'major':            torch.from_numpy(major),
            'major_detail':     torch.from_numpy(major_detail),
            'gen_type':         torch.from_numpy(gen_type),
            'gen_subcat':       torch.from_numpy(gen_subcat),
            'gen_term':         torch.from_numpy(gen_term),
            'subject_div':      torch.from_numpy(subject_div),
            'subject_category': torch.from_numpy(subj_cat),
            'difficulty':       torch.from_numpy(difficulty),
            'evaluation':         torch.from_numpy(evaluation),
            'su_yn':            torch.from_numpy(su_yn_arr),
            'resit_yn':         torch.from_numpy(resit_yn_arr),
            'hist_cont_feats':  torch.from_numpy(hist_cont_feats),
        }

        stu_meta = {
            'college':             torch.from_numpy(stu_college),
            'major':               torch.from_numpy(stu_major),
            'major_detail':        torch.from_numpy(stu_major_detail),
            'student_state':       torch.from_numpy(stu_state),
            'second_major':        torch.from_numpy(stu_second_major),
            'third_major':         torch.from_numpy(stu_third_major),
            'minor_major':         torch.from_numpy(stu_minor_major),
            'second_minor_major':  torch.from_numpy(stu_second_minor_major),
            'micro_major':         torch.from_numpy(stu_micro_major),
            'entrance_major_dept': torch.from_numpy(stu_entrance_major_dept),
            'grad_major_dept':     torch.from_numpy(stu_grad_major_dept),
            'transfer_type':       torch.from_numpy(stu_transfer_type)
        }

        return (
            torch.from_numpy(hist_ids),
            torch.from_numpy(hist_terms),
            torch.from_numpy(mask),
            torch.from_numpy(stu_ids),
            torch.from_numpy(tgt_terms),
            torch.from_numpy(labels),
            course_meta,
            stu_meta
        )

# ======================================================================
# 모델 클래스
# ======================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SharedEmbeddings(nn.Module):
    def __init__(self,
                 d_id: int,
                 d_term: int,
                 d_meta: int,
                 num_students, num_courses, num_terms,
                 num_college, num_major, num_major_detail,
                 num_gen_type, num_gen_subcat, num_gen_term,
                 num_subject_div, num_subj_cat,
                 num_su_yn, num_resit_yn,
                 num_student_state,
                 num_second_major, num_third_major,
                 num_minor_major, num_second_minor_major,
                 num_micro_major,
                 num_entrance_major_dept, num_grad_major_dept,
                 num_transfer_type,
                 dim_kw_precomputed,
                 dim_theme_precomputed,
                 num_history_cont_feats, # 👈 이 인자를 추가해주세요.
                 initial_keyword_embeddings_tensor=None,
                 initial_theme_embeddings_tensor=None):
        super().__init__()

        # 인자로 받은 값을 사용하여 임베딩 레이어 정의
        self.student_id_emb = nn.Embedding(num_students + 1, d_id, padding_idx=0)
        self.course_id_emb  = nn.Embedding(num_courses + 1, d_id, padding_idx=0)
        self.term_id_emb    = nn.Embedding(num_terms + 1, d_term, padding_idx=0)
        self.college_emb      = nn.Embedding(num_college + 1, d_meta, padding_idx=0)
        self.major_emb        = nn.Embedding(num_major + 1, d_meta, padding_idx=0)
        self.major_detail_emb = nn.Embedding(num_major_detail+1, d_meta, padding_idx=0)
        self.gen_type_emb     = nn.Embedding(num_gen_type+1,     d_meta, padding_idx=0)
        self.gen_subcat_emb   = nn.Embedding(num_gen_subcat+1,   d_meta, padding_idx=0)
        self.gen_term_emb     = nn.Embedding(num_gen_term+1,     d_meta, padding_idx=0)
        self.subject_div_emb  = nn.Embedding(num_subject_div+1,  d_meta, padding_idx=0)
        self.subj_cat_emb     = nn.Embedding(num_subj_cat+1,     d_meta, padding_idx=0)
        self.su_yn_emb      = nn.Embedding(num_su_yn+1,    d_meta, padding_idx=0)
        self.resit_yn_emb   = nn.Embedding(num_resit_yn+1, d_meta, padding_idx=0)
        self.student_state_emb = nn.Embedding(num_student_state+1, d_meta, padding_idx=0)
        self.second_major_emb        = nn.Embedding(num_second_major+1,        d_meta, padding_idx=0)
        self.third_major_emb         = nn.Embedding(num_third_major+1,         d_meta, padding_idx=0)
        self.minor_major_emb         = nn.Embedding(num_minor_major+1,         d_meta, padding_idx=0)
        self.second_minor_major_emb  = nn.Embedding(num_second_minor_major+1,  d_meta, padding_idx=0)
        self.micro_major_emb         = nn.Embedding(num_micro_major+1,         d_meta, padding_idx=0)
        self.entrance_major_dept_emb = nn.Embedding(num_entrance_major_dept+1, d_meta, padding_idx=0)
        self.grad_major_dept_emb     = nn.Embedding(num_grad_major_dept+1,     d_meta, padding_idx=0)
        self.transfer_type_emb       = nn.Embedding(num_transfer_type+1,       d_meta, padding_idx=0)
        self.cont_proj_history = nn.Linear(num_history_cont_feats, d_meta)
        self.cont_proj_course  = nn.Linear(2, d_meta)

        if initial_keyword_embeddings_tensor is not None:
            self.trainable_keyword_emb = nn.Embedding.from_pretrained(
                initial_keyword_embeddings_tensor, freeze=False
            )
        else:
            self.trainable_keyword_emb = None
        
                # <--- [추가] Pre-trained Theme Embedding ---
        if initial_theme_embeddings_tensor is not None:
            self.trainable_theme_emb = nn.Embedding.from_pretrained(
                initial_theme_embeddings_tensor, freeze=False
            )
        else:
            self.trainable_theme_emb = None


class TermRecTransformer(nn.Module):
    # [수정 1] __init__ 메서드에 d_id, d_meta, d_term 인자 추가
    def __init__(self,
                 shared_emb: nn.Module,
                 num_courses: int,
                 dim_kw_precomputed: int,
                 dim_theme_precomputed: int,
                 num_history_cont_feats: int,
                 d_model: int,
                 nhead: int,
                 d_ff: int,
                 nlayers: int,
                 dropout: float,
                 d_id: int,      # 👈 추가
                 d_meta: int,    # 👈 추가
                 d_term: int,    # 👈 추가
                 use_positional: bool = True):
        super().__init__()
        self.emb = shared_emb
        self.use_pos = use_positional

        self.meta_weights = nn.ParameterDict({
            'college': nn.Parameter(torch.tensor(2.0)),
            'major': nn.Parameter(torch.tensor(2.0)),
            'major_detail': nn.Parameter(torch.tensor(3.0)),
            'gen_type': nn.Parameter(torch.tensor(2.0)),
            'gen_subcat': nn.Parameter(torch.tensor(2.0)),
            'gen_term': nn.Parameter(torch.tensor(2.0)),
        })
        self.student_meta_weights = nn.ParameterDict({
            'college': nn.Parameter(torch.tensor(2.0)),
            'major': nn.Parameter(torch.tensor(2.0)),
            'major_detail': nn.Parameter(torch.tensor(3.0)),
        })

        # [수정 2] 정의되지 않은 전역 변수 대신, 인자로 받은 변수 사용
        self.kw_proj = nn.Linear(dim_kw_precomputed, d_meta) 
        self.theme_proj = nn.Linear(dim_theme_precomputed, d_meta) 
        self.cont_proj_history = nn.Linear(num_history_cont_feats, d_meta)
        self.cont_proj_course = nn.Linear(2, d_meta)

        total_course_feature_dim = d_id + d_term + 14 * d_meta 
        self.course_proj = nn.Linear(total_course_feature_dim, d_model)
        self.student_proj = nn.Linear(d_id + 12 * d_meta, d_model) 
        self.pos_enc = PositionalEncoding(d_model) if use_positional else nn.Identity() 
        
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=d_ff,
                                               dropout=dropout, batch_first=True) # D_MODEL, NHEAD, D_FF, DROPOUT -> 인자 변수
        self.transformer = nn.TransformerEncoder(enc_layer, nlayers) # NLAYERS -> nlayers
        self.attn_pool = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True) # D_MODEL -> d_model
        self.classifier = nn.Linear(d_model, num_courses) # D_MODEL -> d_model
        self.dropout = nn.Dropout(dropout) # DROPOUT -> dropout

    def build_course_vec(self, course_ids, term_ids, m):
        cid_vec  = self.emb.course_id_emb(course_ids)
        term_vec = self.emb.term_id_emb(term_ids)

        college_vec = self.meta_weights['college'] * self.emb.college_emb(m['college'])
        major_vec   = self.meta_weights['major']   * self.emb.major_emb(m['major'])
        major_detail_vec = self.meta_weights['major_detail'] * self.emb.major_detail_emb(m['major_detail'])
        gen_type_vec     = self.meta_weights['gen_type'] * self.emb.gen_type_emb(m['gen_type'])
        gen_subcat_vec   = self.meta_weights['gen_subcat'] * self.emb.gen_subcat_emb(m['gen_subcat'])
        gen_term_vec     = self.meta_weights['gen_term'] * self.emb.gen_term_emb(m['gen_term'])
        subj_div_vec     = self.emb.subject_div_emb(m['subject_div'])
        subj_cat_vec     = self.emb.subj_cat_emb(m['subject_category'])

        cont_course = torch.stack([
            m['difficulty'], m['evaluation']
        ], dim=-1)
        cont_course_vec = self.cont_proj_course(cont_course.float())

        su_yn_vec    = self.emb.su_yn_emb(m['su_yn'])
        resit_yn_vec = self.emb.resit_yn_emb(m['resit_yn'])
        cont_hist_vec = self.cont_proj_history(m['hist_cont_feats'].float())

        if self.emb.trainable_keyword_emb is not None:
            kw_raw = self.emb.trainable_keyword_emb(course_ids).float()
            kw_vec = self.kw_proj(kw_raw)
        else:
            kw_vec = torch.zeros_like(cont_hist_vec)

        # <--- [추가] Theme 벡터 생성 및 Projection ---
        if self.emb.trainable_theme_emb is not None:
            theme_raw = self.emb.trainable_theme_emb(course_ids).float()
            theme_vec = self.theme_proj(theme_raw)
        else:
            theme_vec = torch.zeros_like(cont_hist_vec)

        # <--- [수정] 모든 벡터를 Concatenate ---
        x = torch.cat([
            cid_vec, term_vec,
            college_vec, major_vec, major_detail_vec,
            gen_type_vec, gen_subcat_vec, gen_term_vec,
            subj_div_vec, subj_cat_vec,
            cont_course_vec,
            su_yn_vec, resit_yn_vec,
            cont_hist_vec, 
            kw_vec, 
            theme_vec # Theme 벡터 추가
        ], dim=-1)
        return self.course_proj(x)

    def build_student_vec(self, stu_ids, sm):
        sid_vec = self.emb.student_id_emb(stu_ids)

        college_vec = self.student_meta_weights['college'] * self.emb.college_emb(sm['college'])
        major_vec   = self.student_meta_weights['major']   * self.emb.major_emb(sm['major'])
        major_detail_vec = self.student_meta_weights['major_detail'] * self.emb.major_detail_emb(sm['major_detail'])
        student_state_vec = self.emb.student_state_emb(sm['student_state'])
        
        second_major_vec = self.emb.second_major_emb(sm['second_major'])
        third_major_vec  = self.emb.third_major_emb(sm['third_major'])
        minor_major_vec  = self.emb.minor_major_emb(sm['minor_major'])
        second_minor_vec = self.emb.second_minor_major_emb(sm['second_minor_major'])
        micro_major_vec  = self.emb.micro_major_emb(sm['micro_major'])
        entrance_dept_vec = self.emb.entrance_major_dept_emb(sm['entrance_major_dept'])
        grad_dept_vec    = self.emb.grad_major_dept_emb(sm['grad_major_dept'])
        transfer_vec     = self.emb.transfer_type_emb(sm['transfer_type'])

        x = torch.cat([
            sid_vec,
            college_vec, major_vec, major_detail_vec,
            student_state_vec,
            second_major_vec, third_major_vec,
            minor_major_vec, second_minor_vec, micro_major_vec,
            entrance_dept_vec, grad_dept_vec,
            transfer_vec
        ], dim=-1)
        return self.student_proj(x)

    def forward(self, hist_ids, hist_terms, mask, stu_ids, course_meta, stu_meta=None, return_attn=False):
        e = self.build_course_vec(hist_ids, hist_terms, course_meta)
        if self.use_pos:
            e = self.pos_enc(e)
        out = self.transformer(e, src_key_padding_mask=~mask)

        if stu_meta is None:
            zeros = torch.zeros_like(stu_ids)
            stu_meta = {k: zeros for k in [
                'college','major','major_detail','student_state',
                'second_major','third_major','minor_major','second_minor_major',
                'micro_major','entrance_major_dept','grad_major_dept',
                'transfer_type','admission_type'
            ]}

        q, attn = self.attn_pool(
            self.build_student_vec(stu_ids, stu_meta).unsqueeze(1),
            out, out, key_padding_mask=~mask
        )
        h = self.dropout(q.squeeze(1))
        logits = self.classifier(h)
        return (logits, attn.squeeze(1)) if return_attn else logits