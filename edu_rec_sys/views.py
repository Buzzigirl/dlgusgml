# edu_rec_sys/views.py

from django.shortcuts import render
from .services.recommendation_service import recommendation_service
import json

def recommend_view(request):
    context = {
        'student_id': None,
        'student_data': None,
        'base_recommendations': None,    # 1단계: 대표 분반 추천 (Top 60)
        'filtered_recommendations': None, # 3단계: 교집합 결과
        'filter_options': recommendation_service.get_filter_options(),
        'submitted_filters': {}
    }
    context['major_hierarchy_json'] = json.dumps(context['filter_options'].get('major_hierarchy', {}))
    context['gyoyang_hierarchy_json'] = json.dumps(context['filter_options'].get('gyoyang_hierarchy', {}))
    
    if request.method == 'POST':
        student_id_str = request.POST.get('student_id')
        if not student_id_str or not student_id_str.isdigit():
            context['error'] = "올바른 학생 ID를 입력해주세요."
            return render(request, 'edu_rec_sys/recommend.html', context)

        student_id = int(student_id_str)
        context['student_id'] = student_id
        
        student_data = recommendation_service.get_student_history(student_id)
        if student_data:
            context['student_data'] = student_data
        else:
            context['error'] = f"{student_id} 학생의 정보가 존재하지 않습니다."
            return render(request, 'edu_rec_sys/recommend.html', context)

        # 1단계: "대표 분반" 목록을 항상 가져와서 화면에 표시
        base_recs_df = recommendation_service.predict_top_k_df(student_id)
        if not base_recs_df.empty:
            context['base_recommendations'] = base_recs_df.to_dict('records')

        if 'is_filtering' in request.POST:
            filter_criteria = {
                # --- 전공/교양 필터 ---
                'subject_category': request.POST.getlist('subject_category'),
                'college_name': request.POST.getlist('college_name'),
                'major_name': request.POST.getlist('major_name'),
                'major_detail': request.POST.getlist('major_detail'),
                'general_type_gyoyang': request.POST.getlist('general_type_gyoyang'),
                'general_subcategory_gyoyang': request.POST.getlist('general_subcategory_gyoyang'),
                'general_term_gyoyang': request.POST.getlist('general_term_gyoyang'),
                'etc_type': request.POST.getlist('etc_type'),

                # --- 시간/요일 필터 (복수선택) ---
                'preferred_days': request.POST.getlist('preferred_days'),
                'preferred_periods': request.POST.getlist('preferred_periods'),
                # --- 기타 필터 ---
                'credit': request.POST.getlist('credit'),
                'class_styles': request.POST.getlist('class_styles'),
                'grade_evaluation': request.POST.getlist('grade_evaluation'),
                'grade_eval_methods': request.POST.getlist('grade_eval_methods'),
                'lecture_methods': request.POST.getlist('lecture_methods'),
            }
            # 빈 값(None, '')을 딕셔너리에서 제거하여 서비스에 전달
            cleaned_criteria = {k: v for k, v in filter_criteria.items() if v}
            context['submitted_filters'] = cleaned_criteria

            # [수정] 교집합을 찾는 최종 메서드 호출
            filtered_df = recommendation_service.get_filtered_recommendations(student_id, cleaned_criteria)
            
            context['filtered_recommendations'] = [] if filtered_df.empty else filtered_df.to_dict('records')

    return render(request, 'edu_rec_sys/recommend.html', context)