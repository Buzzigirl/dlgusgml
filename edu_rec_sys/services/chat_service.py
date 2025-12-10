from django.conf import settings
from .recommendation_service import recommendation_service
from openai import OpenAI
import pandas as pd
import json

class ChatService:
    def __init__(self):
        # Initialize OpenAI client with API key from settings
        api_key = os.environ.get("OPENAI_API_KEY") # Ensure this is loaded in settings
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.rec_service = recommendation_service

    def start_chat(self, student_id):
        """
        Initializes the chat session for a student.
        1. Retrieves Top 80 recommendations.
        2. Sets up initial session state.
        3. Generates the first welcome message.
        """
        # 1. Get Top 80
        # Assuming predict_top_k returns dict with 'topN_subj' list
        # We try to request 80, if function supports it, otherwise slice.
        # Check predict_top_k signature in source if needed, but safe to slice result.
        pred_result = self.rec_service.predict_top_k(student_id)
        
        all_top_ids = []
        if pred_result and 'topN_subj' in pred_result:
            all_top_ids = pred_result['topN_subj'][:80] # Ensure max 80

        # 2. Initial State
        state = {
            'step': 'intro', # Current step waiting for user input
            'top_ids': all_top_ids, # List of SUBJTNB
            'filters': {},
            'history': [] # Chat history for context if needed
        }

        # 3. Generate Message (Static for intro to save tokens or use GPT)
        # Using GPT for persona
        # message = self._get_gpt_response("intro", count=len(all_top_ids))
        # Initial message is standard
        count = len(all_top_ids)
        message = f"안녕하세요! 👋\n학우님의 수강 이력을 분석하여 가장 적합한 **Top {count}** 과목을 찾았습니다.\n\n이 중에서 딱 맞는 과목을 찾아드리기 위해 몇 가지 질문을 드릴게요.\n시작해볼까요?"
        
        choices = [
            {"label": "네, 추천해줘! 🚀", "value": "start_filtering"},
            {"label": "아니오, 전체 결과 볼래요", "value": "show_all"}
        ]
        
        return {
            "message": message,
            "choices": choices,
            "state": state
        }

    def process_message(self, state, user_value):
        """
        Handles user input based on current step.
        Returns new message, choices, updated state.
        """
        step = state.get('step')
        top_ids = state.get('top_ids', [])
        current_filters = state.get('filters', {})
        
        # Determine Next Step & Actions
        next_step = step
        gpt_prompt_type = None
        system_message = ""
        choices = []

        if step == 'intro':
            if user_value == 'show_all':
                return {
                    "message": "알겠습니다! 전체 추천 결과를 표로 보여드릴게요.",
                    "choices": [{"label": "결과 보기", "value": "RESET", "action": "show_table"}],
                    "state": state
                }
            else:
                # Move to Filter 1: Credit
                next_step = 'ask_credit'
                gpt_prompt_type = "ask_credit"
                choices = [
                    {"label": "3학점", "value": "3"},
                    {"label": "2학점", "value": "2"},
                    {"label": "1학점", "value": "1"},
                    {"label": "상관없음", "value": "any"}
                ]

        elif step == 'ask_credit':
            # Apply Credit Filter
            if user_value != 'any':
                current_filters['CDT'] = float(user_value) # Store as float to match DB
            
            # Apply filter logic to update top_ids (count only for now)
            filtered_ids = self._apply_filters(top_ids, current_filters)
            state['top_ids'] = filtered_ids
            
            # Move to Filter 2: Evaluation
            next_step = 'ask_eval'
            gpt_prompt_type = "ask_eval"
            choices = [
                {"label": "상대평가", "value": "상대평가"},
                {"label": "절대평가", "value": "절대평가"},
                {"label": "P/F", "value": "P/F"},
                {"label": "상관없음", "value": "any"}
            ]

        elif step == 'ask_eval':
            # Apply Eval Filter
            # Database columns: GRADE_EVL_MTHD_DIV_CD1 (상대평가, 절대평가, P/F)
            if user_value != 'any':
                current_filters['GRADE_EVL_MTHD_DIV_CD1'] = user_value
            
            filtered_ids = self._apply_filters(top_ids, current_filters)
            state['top_ids'] = filtered_ids
            
            # Move to Final Result
            next_step = 'conclusion'
            gpt_prompt_type = "conclusion"
            choices = [
                {"label": "결과 확인하기 ✨", "value": "show_table", "action": "show_table"}
            ]

        # Update State
        state['step'] = next_step
        state['filters'] = current_filters

        # Generate GPT Response
        rem_count = len(state['top_ids'])
        msg_text = self._generate_gpt_response(gpt_prompt_type, rem_count)

        return {
            "message": msg_text,
            "choices": choices,
            "state": state
        }

    def _apply_filters(self, top_ids, filters):
        # Retrieve Dataframe subset
        # optimize: We need to look up metadata for these IDs.
        # usage: self.rec_service.list_sub_24 row lookup
        
        full_df = self.rec_service.list_sub_24
        
        # Filter 1: Subset to current top_ids
        # Ensure SUBJTNB match (str vs str)
        # full_df['SUBJTNB'] is object/str usually. top_ids are from predict_top_k (might be int or str)
        # Let's align to string
        target_ids = [str(x).strip() for x in top_ids]
        subset = full_df[full_df['SUBJTNB'].astype(str).str.strip().isin(target_ids)]
        
        # Apply Logic
        if 'CDT' in filters:
            # subset['CDT'] might be string "3" or int 3.
            # safe conversion
            try:
                subset = subset[pd.to_numeric(subset['CDT'], errors='coerce') == filters['CDT']]
            except:
                pass
                
        if 'GRADE_EVL_MTHD_DIV_CD1' in filters:
            val = filters['GRADE_EVL_MTHD_DIV_CD1']
            if 'GRADE_EVL_MTHD_DIV_CD1' in subset.columns:
                 subset = subset[subset['GRADE_EVL_MTHD_DIV_CD1'] == val]
        
        return subset['SUBJTNB'].tolist()

    def _generate_gpt_response(self, prompt_type, count):
        if not self.client:
            # Fallback if no API key
            if prompt_type == 'ask_credit': return f"현재 {count}개 과목이 남았습니다. 선호하는 학점이 있으신가요?"
            if prompt_type == 'ask_eval': return f"{count}개로 추려졌습니다! 평가 방식은요?"
            if prompt_type == 'conclusion': return f"최종적으로 {count}개 과목을 찾았습니다. 확인해보시겠어요?"
            return "..."

        system_prompt = "You are a helpful academic advisor chatbot. Be concise, friendly, and encouraging. Use emojis."
        user_prompt = ""
        
        if prompt_type == "ask_credit":
            user_prompt = f"We have {count} recommended courses. Ask the student about their preferred credit points (3, 2, 1, or Any). Keep it short."
        elif prompt_type == "ask_eval":
            user_prompt = f"We narrowed it down to {count} courses based on credits. Now ask about grading method preference (Relative, Absolute, or P/F). Keep it short."
        elif prompt_type == "conclusion":
            user_prompt = f"We found {count} perfect matches. Invite the student to view the results table. Congratulate them."

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"GPT Error: {e}")
            return f"남은 과목: {count}개. 다음 질문으로 넘어갈까요?"

import os
