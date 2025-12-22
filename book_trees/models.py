from django.db import models
from django.utils import timezone

# Create your models here.

class EpubFile(models.Model):
    file = models.FileField(upload_to='epubs/%Y/%m/%d')

    original_filename = models.CharField(max_length=255)

    uploaded_at = models.DateTimeField(default=timezone.now)
    processed = models.BooleanField(default=False)

    STATUS_CHOICES = [
        ('p', 'pending'),
        ('pr', 'processing'),
        ('c', 'completed'),
        ('f', 'failed'),
    ]
    status = models.CharField(max_length=2, choices=STATUS_CHOICES, default='p')

    class Meta:
        ordering = ['uploaded_at']

    def __str__(self):
        return f"{self.original_filename} - {self.uploaded_at.strftime('%Y-%m-%d')}"

class Chapter(models.Model):
    epub = models.ForeignKey(EpubFile, on_delete=models.CASCADE, related_name='chapters')
    title = models.CharField(max_length=500, blank=True)
    content = models.TextField()
    chapter_number = models.IntegerField()
    annotated_sentences = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ['epub', 'chapter_number']
        unique_together = ['epub', 'chapter_number']

    def __str__(self):
        return f"{self.epub.original_filename} - Chapter {self.chapter_number}"


class Character(models.Model):
    epub = models.ForeignKey(EpubFile, on_delete=models.CASCADE, related_name='characters')
    name = models.CharField(max_length=200)
    aliases = models.JSONField(default=list)
    mention_count = models.IntegerField(default=0)
    first_appearance_chapter = models.IntegerField(null=True)

    class Meta:
        ordering = ['-mention_count']
        unique_together = ['epub', 'name']

    def __str__(self):
        return f"{self.name} {self.epub.original_filename}"



class Relationship(models.Model):
    epub = models.ForeignKey(EpubFile, on_delete=models.CASCADE)
    character_1 = models.ForeignKey(Character, on_delete=models.CASCADE, related_name='relationships_as_character_1')
    character_2 = models.ForeignKey(Character, on_delete=models.CASCADE, related_name='relationshios_as_character_2')
    relationship_type = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    supporting_text = models.TextField(blank=True)
    chapter_found = models.IntegerField(null=True, blank=True)

    class Meta:
        unique_together = ['character_1', 'character_2', 'relationship_type']

    def __str__(self):
        return f"{self.character_1.name} - {self.relationship_type} - {self.character_2.name}"
