from django.db import models

# Create your models here.

class Document(models.Model):
    title = models.CharField(max_length=255)
    pdf_file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    chunk_count = models.IntegerField(default=0)
    
    def __str__(self):
        return self.title
