from django import forms
from .models import Document


class DocumentUploadForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ['title', 'pdf_file']

class QueryForm(forms.Form):
    question = forms.CharField(
        widget=forms.Textarea(attrs={
            'placeholder': 'Ask a question about the uploaded document...',
            'rows': 3
        }),
        max_length=1000
        )