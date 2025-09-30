from django.shortcuts import render, redirect
from django.http import JsonResponse
from .forms import DocumentUploadForm, QueryForm
from .models import Document
from .rag_engine import RAGEngine
import os

# Create your views here.

# RAG engine instance
rag_engine = RAGEngine()

def upload_document(request):
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save()

            try:
                chunk_count = rag_engine.process_pdf(document.pdf_file)
                document.processed = True
                document.chunk_count = chunk_count
                document.save()

                return redirect('query_document')
            except Exception as e:
                document.delete()
                form.add_error('pdf_file',f'Error processing PDF: {str(e)}')
    
    else:
        form = DocumentUploadForm()

    return render(request, 'documents/upload.html', {'form': form})


def query_document(request):
    if request.method == 'POST':
        form = QueryForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['question']

            answer = rag_engine.query_document(question)

            similar_chunks = rag_engine.get_similar_chunks(question)

            return render(request, 'documents/query.html', {
                'form': form,
                'question': question,
                'answer': answer,
                'similar_chunks': similar_chunks
            })
        
    else:
        form = QueryForm()

    return render(request, 'documents/query.html', {'form': form})

def api_query(request):
    if request.method == 'POST':
        question = request.POST.get('question', '')

        if question:
            answer = rag_engine.query_document(question)
            return JsonResponse({'question': question, 'answer': answer})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)