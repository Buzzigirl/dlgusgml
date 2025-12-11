# edu_rec_sys/views.py

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .services.recommendation_service import recommendation_service
from .services.chat_service import ChatService
import json
import logging
import traceback

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def recommend_view(request):
    try:
        logger.info(f"📥 Recommendation view called - Method: {request.method}")
        
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
            logger.info(f"📝 Student ID received: {student_id_str}")
            
            if not student_id_str or not student_id_str.isdigit():
                context['error'] = "올바른 학생 ID를 입력해주세요."
                logger.warning(f"⚠️ Invalid student ID format: {student_id_str}")
                return render(request, 'edu_rec_sys/recommend.html', context)

            student_id = int(student_id_str)
            context['student_id'] = student_id
            
            student_data = recommendation_service.get_student_history(student_id)
            if student_data:
                context['student_data'] = student_data
                logger.info(f"✅ Student data loaded for ID: {student_id}")
            else:
                context['error'] = f"{student_id} 학생의 정보가 존재하지 않습니다."
                logger.error(f"❌ Student not found: {student_id}")
                return render(request, 'edu_rec_sys/recommend.html', context)

            # 1단계: "대표 분반" 목록을 항상 가져와서 화면에 표시
            base_recs_df = recommendation_service.predict_top_k_df(student_id)
            if not base_recs_df.empty:
                context['base_recommendations'] = base_recs_df.to_dict('records')
                logger.info(f"📊 Base recommendations generated: {len(base_recs_df)} courses")

            if 'is_filtering' in request.POST:
                logger.info("🔍 Filter criteria received")
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
                logger.info(f"🎯 Applied filters: {list(cleaned_criteria.keys())}")

                # [수정] 교집합을 찾는 최종 메서드 호출
                filtered_df = recommendation_service.get_filtered_recommendations(student_id, cleaned_criteria)
                
                context['filtered_recommendations'] = [] if filtered_df.empty else filtered_df.to_dict('records')
                logger.info(f"✅ Filtered results: {len(context['filtered_recommendations'])} courses")

        logger.info("🎨 Rendering template...")
        return render(request, 'edu_rec_sys/recommend.html', context)
    
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in recommend_view:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        # Return a simple error page
        error_context = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }
        return JsonResponse(error_context, status=500)

# --- Chatbot API Views ---

def start_chat_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            student_id = int(data.get('student_id'))
            
            chat_service = ChatService()
            response_data = chat_service.start_chat(student_id)
            
            # Save state to session
            request.session['chat_state'] = response_data['state']
            
            return JsonResponse({
                'message': response_data['message'],
                'choices': response_data['choices']
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid method'}, status=405)

def chat_message_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_value = data.get('value')
            
            state = request.session.get('chat_state')
            if not state:
                return JsonResponse({'error': 'No active chat session'}, status=400)
            
            chat_service = ChatService()
            response_data = chat_service.process_message(state, user_value)
            
            # Update session
            request.session['chat_state'] = response_data['state']
            
            return JsonResponse({
                'message': response_data['message'],
                'choices': response_data['choices'],
                'action': response_data.get('choices', [{}])[0].get('action') # Helper for frontend to know if done
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid method'}, status=405)
